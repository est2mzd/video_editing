"""
関数実行適性に基づく target 正規化

各関数（GroundingDINO, cv2等）が実際に処理可能な target 形式に統一。
GT の抽象ラベルを関数が検出可能な具体物に変換。
"""

import re
from typing import Optional, Dict, Tuple


def extract_effect_type(instruction: str) -> Optional[str]:
    """
    instruction から effect フレーズを抽出（add_effect 専用）
    
    パターン: `add X effect`, `apply X`, `enhance with X`, `apply X to ...`
    
    Args:
        instruction: 処理指示文
    
    Returns:
        effect 名（glow, lighting, motion_blur, fog 等）、または None
    
    Examples:
        "add glow effect to the person" → "glow"
        "apply stage lighting to the background" → "lighting"
        "enhance the scene with motion blur" → "motion_blur"
    """
    if not instruction:
        return None
    
    instruction_lower = instruction.lower()
    
    # パターン定義：effect 種類と対応する正規化名
    effect_patterns = [
        # 明示的な effect 名パターン
        (r'(?:add|apply|apply|enhance with)\s+(\w+)\s+effect', lambda m: m.group(1)),
        (r'(?:add|apply)\s+(\w+(?:\s+\w+)*?)\s+(?:effect|lighting|glow|blur)', lambda m: m.group(1).replace(' ', '_')),
        # stage lighting パターン
        (r'stage\s+lighting', lambda m: 'lighting'),
        (r'stage\s+light', lambda m: 'lighting'),
        # 個別効果パターン
        (r'glow\s+effect', lambda m: 'glow'),
        (r'motion\s+blur', lambda m: 'motion_blur'),
        (r'bloom\s+effect', lambda m: 'bloom'),
        (r'fog(?:\s+effect)?', lambda m: 'fog'),
        (r'rain(?:\s+effect)?', lambda m: 'rain'),
        (r'particle', lambda m: 'particle'),
        (r'lens\s+flare', lambda m: 'lens_flare'),
        (r'chromatic\s+aberration', lambda m: 'chromatic_aberration'),
        (r'vignette', lambda m: 'vignette'),
    ]
    
    for pattern, extract_fn in effect_patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            effect_name = extract_fn(match)
            return effect_name
    
    # フォールバック：見つからない場合
    return None


def normalize_for_groundingdino(target: str) -> str:
    """
    GroundingDINO 検出可能な具体物に target を統一
    
    複合物体や抽象ラベルを検出可能な具体表現に変換
    
    Args:
        target: GT の target ラベル
    
    Returns:
        正規化された target
    """
    if not target:
        return target
    
    target_lower = target.lower().strip()
    
    # 複数部位を単数物体に統一
    if 'armchair_left' in target_lower or 'armchair_right' in target_lower:
        return 'armchair'
    
    # 人物語を単一表現に統一
    person_words = ['man', 'woman', 'boy', 'girl', 'person', 'face', 'head', 'body', 'his', 'her', 'subject']
    if any(word in target_lower for word in person_words):
        return 'person'
    
    # 抽象ラベルの条件付き処理
    if 'camera_view' in target_lower or 'entire_video' in target_lower or 'entire_frame' in target_lower:
        return 'entire_frame'
    
    if 'stage_lighting_region' in target_lower or 'stage_light' in target_lower:
        return 'stage'
    
    if 'upper_scene' in target_lower or 'upper_part' in target_lower or 'background_area' in target_lower:
        return 'background'
    
    # デフォルト：そのまま返す
    return target


def normalize_for_cv2_color(target: str, color_value: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    cv2 色処理対応 target への変換
    
    複数色指定を単一色に統一し、標準色への変換を試行
    
    Args:
        target: GT の target ラベル
        color_value: 色指定値（英文または RGB値）
    
    Returns:
        (正規化 target, 標準色名)
    
    Examples:
        ("armchair_left", "red") → ("armchair", "red")
        ("object", "pastel blue") → ("object", "blue")
    """
    target_normalized = target.lower().strip()
    
    # 複数対象を統合（armchair_left/right → armchair）
    if 'left' in target_normalized and 'right' in target_normalized:
        target_normalized = target_normalized.split('_')[0]
    elif 'left' in target_normalized or 'right' in target_normalized:
        target_normalized = target_normalized.replace('_left', '').replace('_right', '')
    
    # 標準色への動的マッピング
    standard_colors = {
        'red': 'red',
        'green': 'green',
        'blue': 'blue',
        'yellow': 'yellow',
        'orange': 'orange',
        'purple': 'purple',
        'pink': 'pink',
        'cyan': 'cyan',
        'magenta': 'magenta',
        'white': 'white',
        'black': 'black',
        'gray': 'gray',
        'grey': 'gray',
    }
    
    normalized_color = None
    if color_value:
        color_lower = color_value.lower().strip()
        # 複雑な色説明から中心色を抽出（例: "pastel blue" → "blue"）
        for std_color in standard_colors.keys():
            if std_color in color_lower:
                normalized_color = standard_colors[std_color]
                break
        # 見つからない場合は最初の単語を使用
        if not normalized_color:
            first_word = color_lower.split()[0]
            normalized_color = standard_colors.get(first_word, first_word)
    
    return target_normalized, normalized_color


def normalize_for_function(
    action: str,
    target: str,
    instruction: str = "",
    color_value: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    action 別に適切な target 正規化を実施
    
    Args:
        action: 処理 action 名
        target: GT の target ラベル
        instruction: 処理指示文
        color_value: 色指定値（change_color 対応）
    
    Returns:
        {
            'original_target': 元の target,
            'normalized_target': 正規化後 target,
            'effect_type': effect 名（add_effect の場合のみ）,
            'color_name': 標準色名（change_color の場合のみ）
        }
    """
    result = {
        'original_target': target,
        'normalized_target': target,
        'effect_type': None,
        'color_name': None,
    }
    
    action_lower = action.lower().strip()
    
    # add_effect: instruction から effect 種類を抽出、target は参照しない
    if action_lower == 'add_effect':
        effect_type = extract_effect_type(instruction)
        result['effect_type'] = effect_type or 'unknown'
        # target は使わない（effect 名が primary）
        result['normalized_target'] = None
    
    # change_color: 複数対象を統合、色を標準化
    elif action_lower == 'change_color':
        norm_target, norm_color = normalize_for_cv2_color(target, color_value)
        result['normalized_target'] = norm_target
        result['color_name'] = norm_color
    
    # zoom_in: GroundingDINO 検出可能に統一
    elif action_lower == 'zoom_in':
        result['normalized_target'] = normalize_for_groundingdino(target)
    
    # replace_background: 前景プロンプト（GroundingDINO 検出可能）に統一
    elif action_lower == 'replace_background':
        result['normalized_target'] = normalize_for_groundingdino(target)
    
    # add_object, remove_object: 具体物に統一
    elif action_lower in ['add_object', 'remove_object']:
        result['normalized_target'] = normalize_for_groundingdino(target)
    
    # apply_style: 全体 or 人物か確認
    elif action_lower == 'apply_style':
        target_lower = target.lower().strip()
        if 'full_frame' in target_lower or 'entire_frame' in target_lower:
            result['normalized_target'] = 'full_frame'
        elif any(word in target_lower for word in ['person', 'man', 'woman', 'body']):
            result['normalized_target'] = 'person'
        else:
            result['normalized_target'] = target
    
    # その他: GroundingDINO 検出可能に統一（汎用）
    else:
        result['normalized_target'] = normalize_for_groundingdino(target)
    
    return result


# === テスト用エントリーポイント ===
if __name__ == '__main__':
    # add_effect テスト
    print("=== add_effect ===")
    test_cases_effect = [
        "add glow effect to the person",
        "apply stage lighting to the background",
        "enhance the scene with motion blur",
    ]
    for instr in test_cases_effect:
        effect = extract_effect_type(instr)
        print(f"{instr} → effect: {effect}")
    
    # normalize_for_groundingdino テスト
    print("\n=== normalize_for_groundingdino ===")
    test_cases_gd = [
        "armchair_left",
        "his body",
        "camera_view",
        "stage_lighting_region",
    ]
    for target in test_cases_gd:
        normalized = normalize_for_groundingdino(target)
        print(f"{target} → {normalized}")
    
    # normalize_for_cv2_color テスト
    print("\n=== normalize_for_cv2_color ===")
    test_cases_color = [
        ("armchair_left", "red"),
        ("object", "pastel blue"),
    ]
    for target, color in test_cases_color:
        norm_target, norm_color = normalize_for_cv2_color(target, color)
        print(f"{target}, {color} → {norm_target}, {norm_color}")
    
    # normalize_for_function テスト
    print("\n=== normalize_for_function ===")
    result_add_effect = normalize_for_function('add_effect', 'stage_lighting_region', 'apply stage lighting to the background')
    print(f"add_effect → {result_add_effect}")
    
    result_change_color = normalize_for_function('change_color', 'armchair_left', color_value='red')
    print(f"change_color → {result_change_color}")
    
    result_zoom_in = normalize_for_function('zoom_in', 'camera_view')
    print(f"zoom_in → {result_zoom_in}")
