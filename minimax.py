from typing import Dict, List, Set, Tuple, Optional

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

HEAD_TO_HEAD_PENALTY = 10_000
HUNGRY_THRESHOLD = 50
# Глубина дерева поиска для Minimax
MAX_DEPTH = 3


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
    """
    Основной входной хендлер Battlesnake.
    Переключается между быстрым эвристическим режимом (на первом ходу)
    и полноценным поиском Minimax.
    """
    you = game_state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    
    # На самом первом ходу тело может отсутствовать или состоять из одной головы
    if len(you.get("body", [])) <= 2 or MAX_DEPTH == 0:
        return choose_move_heuristic(game_state)[0]

    best_move, _ = minimax(
        state=game_state,
        depth=MAX_DEPTH,
        maximizing_player=True,
        alpha=float("-inf"),
        beta=float("inf")
    )
    
    # Fallback на случай, если все ветки оказались тупиковыми
    return best_move or choose_move_heuristic(game_state)[0]


def minimax(state: Dict, depth: int, maximizing_player: bool, alpha: float, beta: float) -> Tuple[Optional[str], float]:
    """
    Рекурсивная реализация алгоритма Minimax с альфа-бета отсечением.
    Возвращает лучший ход и его оценку.
    """
    you_id = state["you"]["id"]
    my_length = state["you"]["length"]
    
    # Базовый случай: достигли лимита глубины или игра окончена для нашей змеи
    if depth == 0 or is_dead(state["you"], state["board"]):
        return None, evaluate_state(state, you_id, my_length)

    valid_moves = get_valid_moves(state["you"], state["board"])
    if not valid_moves:
        return None, float("-inf") if maximizing_player else float("inf")

    best_move = None

    if maximizing_player:
        max_eval = float("-inf")
        for move in valid_moves:
            next_state = simulate_step(state, you_id, move)
            _, eval_score = minimax(next_state, depth - 1, False, alpha, beta)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Бета-отсечение
        return best_move, max_eval
    else:
        min_eval = float("inf")
        
        # Определяем врагов (для простоты считаем, что минимизирующий игрок — это самый длинный враг)
        enemies = [s for s in state["board"]["snakes"] if s["id"] != you_id]
        if not enemies:
            return best_move, evaluate_state(state, you_id, my_length)
        
        # Сортируем по длине, чтобы сильнейший враг делал ход первым
        enemy = max(enemies, key=lambda x: x["length"])
        
        for move in get_valid_moves(enemy, state["board"]):
            next_state = simulate_step(state, enemy["id"], move)
            _, eval_score = minimax(next_state, depth - 1, True, alpha, beta)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Альфа-отсечение
                
        return best_move, min_eval


def choose_move_heuristic(game_state: Dict, direction: str = 'max') -> Tuple[str, float]:
    """Урезанная эвристика для fallback-режима и оценки листьев дерева."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf") if direction == 'max' else float("inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height) or nxt in occupied:
            continue

        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 5)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if direction == 'max':
            if score > best_score:
                best_score, best_move = score, move
        else:
            if score < best_score:
                best_score, best_move = score, move

    return best_move or "up", best_score


def simulate_step(state: Dict, acting_snake_id: str, move: str) -> Dict:
    """
    Создает глубокую копию состояния и применяет к указанной змее один ход.
    Еда съедается, хвосты сдвигаются автоматически согласно правилам API.
    """
    new_state = deep_copy_state(state)
    snakes = {s["id"]: s for s in new_state["board"]["snakes"]}
    
    if acting_snake_id not in snakes:
        return new_state
        
    snake = snakes[acting_snake_id]
    head = {"x": snake["head"]["x"], "y": snake["head"]["y"]}
    dx, dy = DIRECTIONS[move]
    
    new_head = {"x": head["x"] + dx, "y": head["y"] + dy}
    
    # Проверяем еду до обновления тела
    ate_food = any(f["x"] == new_head["x"] and f["y"] == new_head["y"] for f in new_state["board"]["food"])
    
    # Обновляем позицию
    old_body = list(snake["body"])
    snake["body"].insert(0, new_head)
    snake["head"] = new_head
    
    if not ate_food:
        # Если еды нет — убираем хвост (API сам присылает обновленные body со сдвинутыми хвостами,
        # но при симуляции мы должны делать это вручную)
        if len(snake["body"]) > 1:
            snake["body"].pop()
            
    # Если съели еду — удаляем её с поля
    if ate_food:
        new_state["board"]["food"] = [
            f for f in new_state["board"]["food"] 
            if not (f["x"] == new_head["x"] and f["y"] == new_head["y"])
        ]
        
    return new_state


def deep_copy_state(state: Dict) -> Dict:
    """Глубокое копирование только тех полей, которые нужны для расчетов."""
    def copy_point(p): return {"x": p["x"], "y": p["y"]}
    
    new_you = {
        "id": state["you"]["id"],
        "length": state["you"]["length"],
        "health": state["you"]["health"],
        "head": copy_point(state["you"]["head"]),
        "body": [copy_point(seg) for seg in state["you"]["body"]]
    }
    
    new_snakes = []
    for s in state["board"]["snakes"]:
        new_snakes.append({
            "id": s["id"],
            "length": s["length"],
            "health": s["health"],
            "head": copy_point(s["head"]),
            "body": [copy_point(seg) for seg in s["body"]]
        })
        
    new_food = [copy_point(f) for f in state["board"]["food"]]
        
    return {
        "turn": state["turn"],
        "board": {
            "height": state["board"]["height"],
            "width": state["board"]["width"],
            "snakes": new_snakes,
            "food": new_food
        },
        "you": new_you
    }


def get_valid_moves(snake: Dict, board: Dict) -> List[str]:
    """Возвращает список ходов, которые не приведут к мгновенному столкновению со стеной или телом."""
    head = (snake["head"]["x"], snake["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    width, height = board["width"], board["height"]
    
    valid = []
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in occupied:
            valid.append(move)
    return valid


def is_dead(snake: Dict, board: Dict) -> bool:
    """Проверка смерти змеи по правилам Battlesnake (стены, другие змеи)."""
    head = (snake["head"]["x"], snake["head"]["y"])
    if not _in_bounds(head, board["width"], board["height"]):
        return True
    
    all_bodies = set()
    for other in board["snakes"]:
        for seg in other["body"]:
            all_bodies.add((seg["x"], seg["y"]))
            
    return head in all_bodies


def evaluate_state(state: Dict, my_id: str, my_length: int) -> float:
    """
    Функция оценки (статическая оценка) листа дерева.
    Учитывает выживаемость, длину и количество доступной еды.
    """
    me = next((s for s in state["board"]["snakes"] if s["id"] == my_id), None)
    if me is None or is_dead(me, state["board"]):
        return float("-inf")
        
    score = float(len(me["body"]))
    
    # Бонус за еду на поле
    score += len(state["board"]["food"]) * 5
    
    # Штраф, если мы короче самого длинного врага
    longest_enemy_len = max([s["length"] for s in state["board"]["snakes"] if s["id"] != my_id] + [0])
    if my_length < longest_enemy_len:
        score -= (longest_enemy_len - my_length) * 10
        
    return score


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
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


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
