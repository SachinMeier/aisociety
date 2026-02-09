"""Hierarchical policy/value bot backed by a PyTorch model."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.domain.rules import GameResult
from highsociety.ml.checkpoints import load_checkpoint
from highsociety.ml.encoders.basic import BasicEncoder

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from highsociety.ml.models.hierarchical import HierarchicalConfig, HierarchicalPolicyValue


# Money card values in sorted order (must match model)
MONEY_CARD_VALUES: tuple[int, ...] = (1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000)


@dataclass
class HierarchicalBot:
    """Bot that selects actions using a trained hierarchical policy/value model."""

    name: str = "hierarchical"
    checkpoint: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    device: str = "cpu"
    kind: str = "hierarchical"
    _rng: random.Random = field(init=False)
    _player_id: int | None = None
    _encoder: BasicEncoder = field(init=False, repr=False)
    _model: "HierarchicalPolicyValue" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the model checkpoint and initialize RNG."""
        self._rng = random.Random(self.seed)
        if self.checkpoint is None:
            raise ValueError("checkpoint is required for HierarchicalBot")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        self._load_checkpoint(self.checkpoint)

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        """Reset the bot for a new game, reseeding if needed."""
        self._player_id = player_id
        seed_value = self.seed
        if seed_value is None:
            config_seed = game_config.get("seed")
            if isinstance(config_seed, int):
                seed_value = config_seed
        if seed_value is not None:
            self._rng = random.Random(seed_value + player_id)

    def act(self, observation: Observation, legal_actions: list[Action]) -> Action:
        """Select an action using the hierarchical policy."""
        if not legal_actions:
            raise InvalidAction("No legal actions available")
        if self._player_id is None:
            raise InvalidState("Player has not been reset")

        # Fast path: only one legal action
        if len(legal_actions) == 1:
            return legal_actions[0]

        import torch

        # Encode observation
        features = torch.tensor(
            self._encoder.encode(observation),
            dtype=torch.float32,
            device=self.device,
        )

        # Forward pass
        with torch.no_grad():
            type_logits, card_probs, discard_logits, _ = self._model(features)

        type_logits = type_logits.squeeze(0)
        card_probs = card_probs.squeeze(0)
        discard_logits = discard_logits.squeeze(0)

        # Build masks
        type_mask, card_mask, discard_mask = self._build_masks(legal_actions)

        # Sample action type
        masked_type_logits = type_logits.clone()
        masked_type_logits[~type_mask] = float("-inf")

        if not torch.any(type_mask):
            raise InvalidState("No legal action types available")

        if self.temperature <= 0:
            action_type_idx = int(torch.argmax(masked_type_logits).item())
        else:
            probs = torch.softmax(masked_type_logits / self.temperature, dim=0)
            action_type_idx = _sample_index(self._rng, probs.cpu().tolist())

        # Based on action type, sample parameters
        if action_type_idx == 0:  # PASS
            return self._find_pass_action(legal_actions)

        elif action_type_idx == 1:  # BID
            return self._sample_bid_action(card_probs, card_mask, legal_actions)

        else:  # DISCARD (action_type_idx == 2)
            masked_discard_logits = discard_logits.clone()
            masked_discard_logits[~discard_mask] = float("-inf")

            if not torch.any(discard_mask):
                # Fallback to first discard action
                for action in legal_actions:
                    if action.kind == ActionKind.DISCARD_POSSESSION:
                        return action
                return legal_actions[0]

            if self.temperature <= 0:
                possession_idx = int(torch.argmax(masked_discard_logits).item())
            else:
                probs = torch.softmax(masked_discard_logits / self.temperature, dim=0)
                possession_idx = _sample_index(self._rng, probs.cpu().tolist())

            return self._find_discard_action(legal_actions, possession_idx + 1)

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler."""

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model + encoder configuration from a checkpoint."""
        bundle = load_checkpoint(checkpoint_path)
        encoder = BasicEncoder.from_config(bundle.encoder_config)
        model_config = _load_model_config(bundle.model_config)
        if model_config.input_dim != encoder.feature_size:
            raise InvalidState("Encoder feature size does not match model config")
        model = _load_model(model_config, bundle.model_state, self.device)
        self._encoder = encoder
        self._model = model

    def _build_masks(
        self, legal_actions: list[Action]
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Build action masks from legal actions."""
        import torch

        type_mask = torch.zeros(3, dtype=torch.bool, device=self.device)
        card_mask = torch.zeros(len(MONEY_CARD_VALUES), dtype=torch.bool, device=self.device)
        discard_mask = torch.zeros(10, dtype=torch.bool, device=self.device)

        for action in legal_actions:
            if action.kind == ActionKind.PASS:
                type_mask[0] = True
            elif action.kind == ActionKind.BID:
                type_mask[1] = True
                for card_value in action.cards:
                    try:
                        idx = MONEY_CARD_VALUES.index(card_value)
                        card_mask[idx] = True
                    except ValueError:
                        pass
            elif action.kind == ActionKind.DISCARD_POSSESSION:
                type_mask[2] = True
                if action.possession_value is not None:
                    idx = action.possession_value - 1
                    if 0 <= idx < 10:
                        discard_mask[idx] = True

        return type_mask, card_mask, discard_mask

    def _find_pass_action(self, legal_actions: list[Action]) -> Action:
        """Find the PASS action in legal actions."""
        for action in legal_actions:
            if action.kind == ActionKind.PASS:
                return action
        return legal_actions[0]

    def _find_discard_action(self, legal_actions: list[Action], possession_value: int) -> Action:
        """Find a discard action with the given possession value."""
        for action in legal_actions:
            if action.kind == ActionKind.DISCARD_POSSESSION:
                if action.possession_value == possession_value:
                    return action
        # Fallback to first discard action
        for action in legal_actions:
            if action.kind == ActionKind.DISCARD_POSSESSION:
                return action
        return legal_actions[0]

    def _sample_bid_action(
        self,
        card_probs: "torch.Tensor",
        card_mask: "torch.Tensor",
        legal_actions: list[Action],
    ) -> Action:
        """Sample a bid action from card probabilities."""
        import torch

        # Apply mask to probabilities
        masked_probs = card_probs * card_mask.float()

        # Sample binary decisions for each card
        if self.temperature <= 0:
            # Greedy: select cards with prob > 0.5 (or highest if none)
            selected = (masked_probs > 0.5).float()
            if selected.sum() == 0:
                valid_probs = masked_probs.clone()
                valid_probs[~card_mask] = 0
                if valid_probs.sum() > 0:
                    selected[valid_probs.argmax()] = 1
                else:
                    for i in range(len(card_mask)):
                        if card_mask[i]:
                            selected[i] = 1
                            break
        else:
            # Stochastic sampling
            selected = torch.bernoulli(masked_probs)
            if selected.sum() == 0:
                valid_probs = masked_probs.clone()
                valid_probs[~card_mask] = 0
                if valid_probs.sum() > 0:
                    selected[valid_probs.argmax()] = 1
                else:
                    for i in range(len(card_mask)):
                        if card_mask[i]:
                            selected[i] = 1
                            break

        # Convert selected cards to values
        selected_values = set()
        for i in range(len(MONEY_CARD_VALUES)):
            if selected[i] == 1:
                selected_values.add(MONEY_CARD_VALUES[i])

        # Find matching bid action
        return self._find_best_bid_action(selected_values, legal_actions)

    def _find_best_bid_action(
        self, selected_values: set[int], legal_actions: list[Action]
    ) -> Action:
        """Find the legal bid action that best matches selected card values."""
        bid_actions = [a for a in legal_actions if a.kind == ActionKind.BID]

        if not bid_actions:
            return legal_actions[0]

        # Try exact match first
        for action in bid_actions:
            action_values = set(action.cards)
            if action_values == selected_values:
                return action

        # Find best match by Jaccard similarity
        best_action = bid_actions[0]
        best_score = -1.0

        for action in bid_actions:
            action_values = set(action.cards)
            intersection = len(selected_values & action_values)
            union = len(selected_values | action_values)
            score = intersection / union if union > 0 else 0.0

            # Tie-break: prefer smaller total value (more conservative)
            if score > best_score or (
                score == best_score
                and sum(action.cards) < sum(best_action.cards)
            ):
                best_score = score
                best_action = action

        return best_action


def _load_model_config(data: dict[str, object]) -> "HierarchicalConfig":
    """Load the HierarchicalConfig from a mapping."""
    from highsociety.ml.models.hierarchical import HierarchicalConfig

    return HierarchicalConfig.from_dict(data)


def _load_model(
    config: "HierarchicalConfig",
    state_dict: dict[str, object],
    device: str,
) -> "HierarchicalPolicyValue":
    """Load a HierarchicalPolicyValue model from a state dict."""
    from highsociety.ml.models.hierarchical import HierarchicalPolicyValue

    model = HierarchicalPolicyValue(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _sample_index(rng: random.Random, probs: list[float]) -> int:
    """Sample an index from a probability distribution."""
    total = sum(probs)
    if total <= 0:
        return 0
    threshold = rng.random() * total
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if cumulative >= threshold:
            return idx
    return max(0, len(probs) - 1)
