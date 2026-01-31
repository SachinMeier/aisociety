from highsociety.domain.actions import Action, ActionKind
from highsociety.server import GameServer, PlayerManifestEntry


def make_manifest():
    return [
        PlayerManifestEntry(name="p0", kind="human"),
        PlayerManifestEntry(name="p1", kind="bot"),
        PlayerManifestEntry(name="p2", kind="bot"),
    ]


def test_multiple_games_independent():
    server = GameServer()
    game_a = server.new_game(make_manifest(), seed=1)
    game_b = server.new_game(make_manifest(), seed=2)
    state_b_before = server.get_state(game_b)
    len_before = len(state_b_before.status_deck)
    server.step(game_a, 0, Action(ActionKind.PASS))
    state_b_after = server.get_state(game_b)
    assert len(state_b_after.status_deck) == len_before


def test_invalid_action_does_not_crash_server():
    server = GameServer()
    game_id = server.new_game(make_manifest(), seed=1)
    result = server.step(game_id, 0, Action(ActionKind.BID, cards=(999,)))
    assert result.error is not None
    assert result.fatal is False
    assert server.get_status(game_id) == "active"
