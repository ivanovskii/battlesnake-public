"""Heuristic move-selection logic for the Battlesnake.

This is a deliberately simple, rules-based policy. Everything flows through
:func:`choose_move`, which takes the raw game state and returns one of
``"up" | "down" | "left" | "right"``.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

from typing import Dict, List, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50
# Reward for a move that keeps our own tail reachable (anti-self-trap).
TAIL_REACH_BONUS = 5
# Small per-cell pull away from walls (walls are where we get trapped).
WALL_DIST_WEIGHT = 0.2
# Per-cell reward for keeping distance from equal-or-longer enemy heads. Keeps us
# out of the sandwiches that force losing head-to-heads (multiplayer only).
THREAT_DIST_WEIGHT = 0.5


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the hand-written heuristic."""
    return choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    # Center bias helps when we're alone (don't run out of room at the edges)
    # but hurts against opponents (it pulls both snakes into the same center and
    # into head-to-heads), so only apply it when no enemy snakes are on the board.
    alone = sum(1 for s in board["snakes"] if s["id"] != you["id"]) == 0
    # Heads of enemies at least our length: moving next to these risks a lost
    # head-to-head, so we keep our distance from them (see THREAT_DIST_WEIGHT).
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in board["snakes"]
                    if s["id"] != you["id"] and s["length"] >= my_length]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        # Tail-following: if we can still reach our own tail from here, we can
        # keep coiling without sealing ourselves in (the tail vacates as we
        # move). This breaks the flat-scoring ties that otherwise default to
        # "up" and walk us into a wall.
        if _reaches_tail(nxt, you, occupied, width, height):
            score += TAIL_REACH_BONUS

        # Gentle center bias (solo only): distance to the nearest wall. Keeps us
        # off the edges where space is easiest to run out of. Low weight so it
        # only breaks ties left by space/tail.
        if alone:
            wall_dist = min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
            score += WALL_DIST_WEIGHT * wall_dist

        # Stay away from equal-or-longer enemy heads so we don't get sandwiched
        # into a forced head-to-head. Farther = safer.
        if bigger_heads:
            threat_dist = min(_manhattan(nxt, h) for h in bigger_heads)
            score += THREAT_DIST_WEIGHT * threat_dist

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """All cells currently filled by any snake's body.

    We keep tails occupied too; they only free up *next* turn and treating them
    as solid is the conservative, safe choice for a base bot.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that are >= our length.

    Moving onto one of these risks a head-to-head collision we would lose or
    tie, so they are heavily penalized (but not forbidden — sometimes it's the
    only move).
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Used to avoid moves that would seal us into a small pocket.
    """
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _reaches_tail(start: Point, you: Dict, occupied: Set[Point], width: int, height: int) -> bool:
    """Can we reach our own tail from ``start``?

    The tail cell vacates next turn (unless we eat), so we exclude it from the
    obstacles. If a path exists, moving here won't seal us into a dead pocket.
    """
    tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    if start == tail:
        return True
    free = occupied - {tail}
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr == tail:
                return True
            if nbr in seen or not _in_bounds(nbr, width, height) or nbr in free:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return False


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
