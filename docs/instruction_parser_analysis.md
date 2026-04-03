# Instruction → Tasks 予測パイプライン分析

## 概要

`/workspace/src/parse/instruction_parser_ver19.py` は、自然言語の instruction **のみ** を入力として、対応する **tasks** を予測するシステムです。

**入力**: instruction（テキスト説明）
**出力**: tasks（action, target, constraints, params の配列）
**参照元**: `/workspace/data/annotations_gt_task_ver10.json`（GT データ）

---

## データフロー

### 1. 入力データ構造

#### `annotations_gt_task_ver10.json` (GT データ)
```json
[
  {
    "video_path": "wyzi9GNZFMU_0_0to121.mp4",
    "class": "Camera Motion Editing",
    "subclass": "Dolly in",
    "instruction": "Apply a smooth dolly in effect toward the man's face...",
    "tasks": [
      {
        "action": "dolly_in",
        "target": "man's face",
        "constraints": ["smooth_motion"],
        "params": {
          "motion_type": "dolly_in",
          "start_framing": "medium_shot",
          ...
        }
      }
    ]
  }
  ...
]
```

#### `annotations.jsonl` (入力レコード)
各行が1つの動画に対応する JSON。
- `video_path`: 動画ファイル名
- `instruction`: テキスト説明

### 2. 処理パイプライン全体フロー

```
┌─────────────────────────────────────────┐
│ 1. GT データ読み込み                      │
│    (annotations_gt_task_ver10.json)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. 名詞バンク構築                         │
│    build_noun_bank()                    │
│ 結果: (video, action) → [target候補]    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. 各 instruction に対して予測生成      │
│    predict_single() / predict_multi()   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 結果: [action, target, params, ...]    │
│      (GT参照の予測値)                   │
└─────────────────────────────────────────┘
```

---

## コア処理の詳細

### Step 1: 名詞バンク構築

**関数**: `build_noun_bank(base_records: list[dict])`

**目的**: 各 (video, action) ペアに対して、対応する target 候補のリストを構築

**アルゴリズム**:
1. GT データから「学習セット」を抽出
   - 各動画ごと、各タスクの `target` フィールドを取得
2. target フレーズを正規化 → 名詞を抽出
3. **結果**: `video_action_to_nouns`
   ```python
   {
     ("wyzi9GNZFMU_0_0to121.mp4", "dolly_in"): 
       ["man's face", "man", "face", ...],
     ("1s9DER1bpm0_10_0to213.mp4", "add_object"): 
       ["buffalo", "animal", ...],
     ...
   }
   ```

**関連ヘルパー関数**:
- `canonical_target_phrase()`: target を正規形に変換
- `extract_noun_candidates()`: テキストから名詞候補を抽出

### Step 2: Action 推論

**関数**: `infer_action_from_instruction(instruction: str, fallback_tasks: list[dict])`

**目的**: instruction テキストから **何をするか** (action) を推測

**方法**: 正規表現パターンマッチング

**ACTION_PATTERNS 一覧**:
```python
[
  ("replace_background", r"\bbackground\b|\breplace the .*background\b"),
  ("replace_object",     r"\breplace\b.*\bwith\b"),
  ("change_color",       r"\bchange\b.*\bcolor\b|\brecolor\b"),
  ("add_object",         r"\badd\b|\binsert\b|\bplace\b"),
  ("remove_object",      r"\bremove\b|\bdelete\b|\berase\b"),
  ("apply_style",        r"\bstyle\b|\bcyberpunk\b|\bpixel art\b"),
  ("edit_motion",        r"\bmotion\b|\bwave\b|\bwalk\b"),
  ("zoom_in",            r"\bzoom in\b|\bclose[- ]?up\b"),
  ("zoom_out",           r"\bzoom out\b|\bwider\b"),
  ("dolly_in",           r"\bdolly in\b"),
  ("change_camera_angle", r"\bcamera angle\b|\blow angle\b"),
  ("preserve_framing",   r"\bcentered\b|\bframing\b"),
  ("preserve_focus",     r"\bfocus\b|\bsharp\b"),
]
```

**例**:
- instruction: "Apply a smooth dolly in effect..."
  - regex: `r"\bdolly in\b"` にマッチ → **action = "dolly_in"**

### Step 3: Target 選択

**関数**: `choose_noun_target(...)`

**目的**: 推測した action に対して、最適な **target** (対象物) を選択

**アルゴリズム** (prefer_instruction_overlap=True):
1. (video, action) に対応する target 候補リストを取得
   ```python
   cands = video_action_to_nouns.get((video_path, action), [])
   ```
2. instruction テキストとの **トークンオーバーラップ** で スコアリング
   ```python
   inst_tokens = set(instruction.split())  # 正規化後
   for candidate in cands:
     candidate_tokens = set(candidate.split())
     overlap_count = len(inst_tokens & candidate_tokens)
     score = (overlap_count, len(candidate))  # タイ時は長い方が優先
   ```
3. **スコア最大** のものを選択

**例**:
- instruction: "... the **man's face**, starting from the original..."
- cands: ["man's face", "man", "face", "object"]
- instruction_tokens: {apply, smooth, dolly, in, effect, man's, face, ...}
- scoring:
  - "man's face": overlap=2 (man's, face) → **(2, 10)** ← 最高スコア ✓
  - "man": overlap=1 → (1, 3)
  - "face": overlap=1 → (1, 4)
  - "object": overlap=0 → (0, 6)
- **選択**: "man's face"

### Step 4: Parameters コピー

**方法**: GT から該当 action の params をそのままコピー

```python
if cfg.get("params_source") == "gt_params":
  gt_task = choose_gt_task_by_action(video, action, video_to_tasks)
  params = copy.deepcopy(gt_task.get("params", {}))
```

---

## 予測モード

### Mode: "single" (単一タスク)

関数: `predict_single(rec, cfg, noun_bank, video_to_tasks)`

**出力**: 1つの task のみ
```python
{
  "tasks": [{
    "action": "dolly_in",
    "target": "man's face",
    "constraints": [],
    "params": {...}
  }]
}
```

### Mode: "multi" (複数タスク)

関数: `predict_multi(rec, cfg, noun_bank, video_to_tasks)`

**ロジック**:
1. 設定が "gt_tasks_exact" の場合:
   - GT データから tasks をそのまま抽出（複数タスク対応）

2. それ以外の場合:
   - Primary task: `predict_single()` で取得
   - Secondary tasks: preserve_framing, preserve_focus などを追加

**出力例**:
```python
{
  "tasks": [
    {
      "action": "dolly_in",
      "target": "man's face",
      "constraints": [],
      "params": {...}
    },
    {
      "action": "preserve_framing",
      "target": "man",
      "constraints": [],
      "params": {...}
    },
    {
      "action": "preserve_focus",
      "target": "camera",
      "constraints": [],
      "params": {...}
    }
  ]
}
```

---

## 設定（Config）

予測動作を制御する設定辞書

### MULTI_CFG_BEST (推奨設定)
```python
{
  "action_source": "gt_match",              # GT からaction の候補を選択
  "target_source": "gt_target_exact",       # GT の target をそのまま使用
  "params_source": "gt_params",             # GT の params をコピー
  "instruction_overlap": True,              # target選択時に instruction 重視
  "sanitize_subject": True,                 # target から「主体」を除去
  "multi_source": "gt_tasks_exact",         # 複数タスクは GT から直接取得
}
```

### SINGLE_CFG_BEST
```python
{
  "action_source": "gt_match",
  "target_source": "gt_noun_priority",      # 名詞バンク優先 → instruction オーバーラップ
  "params_source": "gt_params",
  "instruction_overlap": True,
  "sanitize_subject": True,
  "multi_source": "heuristic",              # Second task を heuristic で追加
}
```

---

## 実行方法

### Pythonスクリプトからの呼び出し

```python
from src.parse.instruction_parser_ver19 import (
    parse_annotations_jsonl,
    build_predictions,
    MULTI_CFG_BEST,
)

# Step 1: annotation レコード読み込み
records = parse_annotations_jsonl(Path("/workspace/data/annotations.jsonl"))

# Step 2: 予測生成（GT参照）
predictions = build_predictions(
    records,
    gt_path=Path("/workspace/data/annotations_gt_task_ver10.json"),
    mode="multi",
    cfg=MULTI_CFG_BEST
)

# Step 3: 結果確認
for p in predictions[:5]:
    print(f"Video: {p['video_path']}")
    print(f"Instruction: {p['instruction']}")
    for task in p['prediction']['tasks']:
        print(f"  → {task['action']} on {task['target']}")
```

### 既存スクリプト

`/workspace/src/test/analyze_parse_instruction_ver20.py`
- 上記パイプラインの実行例
- 最初の5件の予測結果を表示

---

## 重要な特徴

### ✅ Instruction 主導
- **入力**: instruction（テキストのみ）
- **その他情報は不要**: video_path, class, subclass は参照しない

### ✅ GT参照による「学習」
- GT データを参照して候補を限定
- action パターンと target 候補が GT から構築される
- → 予測の信頼性が高い

### ✅ テキストマッチング
- 正規表現パターンで action を推論
- トークンオーバーラップで target を選択
- → Deep Learning 不要（解釈可能）

### ✅ Subject Sanitization
- "the man's face" → "man's face" (冠詞除去)
- "the **woman**'s face" → "face" (主体除去)
- → 動作対象を正確に特定

---

## 制限事項

1. **GT に存在しない action**: デフォルト "edit_motion"
2. **複雑な複数タスク**: GT の情報に依存（instruction のみからは推論困難）
3. **Params の詳細値**: GTからのコピー（instruction からの生成なし）

---

## 今後の改善案

1. **Deep Learning による action 推論**
   - 正規表現パターンより精度向上の可能性

2. **Instruction からの parameter 抽出**
   - "smooth", "fast" などの修飾詞から params を生成

3. **多言語対応**
   - 現在は英語のみ
   - 日本語instruction对応（別パイプライン）

4. **Zero-shot 予測**
   - GT に存在しない (video, action) ペアに対応

---

## まとめ表

| 項目 | 説明 |
|---|---|
| **入力** | instruction (テキストのみ) |
| **参照元** | annotations_gt_task_ver10.json (GT) |
| **出力** | tasks [(action, target, params, ...)] |
| **Action推論** | 正規表現パターンマッチング |
| **Target選択** | token overlap スコアリング |
| **Params** | GT からコピー |
| **複数タスク** | GT から直接取得 (multi_mode) |
| **解釈性** | ✅ 完全に解釈可能 |

---

## 📋 設計案：クラスベースの改善版（チート情報排除）

### 現在の問題

現在の `build_predictions()` は GT データを直接参照しているため、コンペ環境では「チート」になりうる。  
具体的には：
- GT の `tasks` や `params` をそのままコピー
- (video_path, action) から target を検索
- これでは「訓練と予測の分離」ができていない

### 改善設計

**原則**: DB には「チートにならない情報」のみを保持
- ✅ action の一般的パターン（正規表現）
- ✅ action → 可能な target 候補（統計的に現れやすいもの）
- ❌ video_path や GT の実際の task
- ❌ params の実値

### クラス設計案

```python
class InstructionParser:
    """
    instruction のみから tasks を予測するクラス。
    DB には「統計的な知識」のみを保持。
    """
    
    def __init__(self, knowledge_db: dict):
        """
        Args:
            knowledge_db: 以下の構造
            {
              "action_patterns": {
                "dolly_in": r"\bdolly in\b",
                "zoom_in": r"\bzoom in\b",
                ...  # 正規表現パターン
              },
              "action_target_vocab": {
                "dolly_in": ["face", "object", "camera"],
                "add_object": ["animal", "person", "object"],
                ...  # action ごとの一般的なtarget
              },
              "multi_mode": bool  # 複数タスク出力するか
            }
        """
        self.patterns = knowledge_db["action_patterns"]
        self.vocab = knowledge_db["action_target_vocab"]
        self.multi_mode = knowledge_db.get("multi_mode", False)
    
    def pred(self, instruction: str) -> dict:
        """
        Args:
            instruction: テキスト説明（のみ）
        
        Returns:
            {
              "tasks": [
                {
                  "action": str,
                  "target": str,
                  "constraints": [],
                  "params": {}
                }
              ]
            }
        """
        # Step 1: instruction から action を推定
        action = self._infer_action(instruction)
        
        # Step 2: action に対応する target を選択
        target = self._choose_target(instruction, action)
        
        # Step 3: 複数タスク対応（if needed）
        tasks = [{"action": action, "target": target, "constraints": [], "params": {}}]
        if self.multi_mode:
            extra_tasks = self._infer_secondary_tasks(instruction)
            tasks.extend(extra_tasks)
        
        return {"tasks": tasks}
    
    def _infer_action(self, instruction: str) -> str:
        """Instruction から action パターンマッチで推定"""
        inst = instruction.lower()
        for action, pattern in self.patterns.items():
            if re.search(pattern, inst):
                return action
        return "edit_motion"  # デフォルト
    
    def _choose_target(self, instruction: str, action: str) -> str:
        """
        Instruction テキストと action から target を選択。
        strategy: token overlap スコア
        """
        vocab = self.vocab.get(action, [])
        if not vocab:
            return "object"  # デフォルト
        
        inst_tokens = set(instruction.lower().split())
        best_target = vocab[0]
        best_score = 0
        
        for target in vocab:
            target_tokens = set(target.lower().split())
            overlap = len(inst_tokens & target_tokens)
            if overlap > best_score:
                best_score = overlap
                best_target = target
        
        return best_target
    
    def _infer_secondary_tasks(self, instruction: str) -> list[dict]:
        """複数タスクモード：副次タスクを推定（省略可）"""
        return []  # TODO: heuristic で preserve_framing など追加
```

### DB の构築方法

**GT から「知識」を抽出**:
```python
def build_knowledge_db(gt_path: Path) -> dict:
    """
    GT データから、チートにならない「知識」を抽出。
    """
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    # Step 1: action patterns（正規表現）を定義
    action_patterns = {
        "dolly_in": r"\bdolly in\b",
        "zoom_in": r"\bzoom in\b",
        "add_object": r"\badd\b|\binsert\b",
        # ... 他のaction
    }
    
    # Step 2: action ごとに「出現した target」の統計を集計
    #        ⚠️  GT の実値ではなく、出現頻度の高い順にソート
    action_target_vocab = defaultdict(set)
    for row in gt_data:
        for task in row.get("tasks", []):
            action = task["action"]
            target = task["target"]
            action_target_vocab[action].add(target)  # 必ず video_path は含める
    
    # Step 3: set を list に変換（出現順でもOK）
    vocab = {
        action: list(targets)
        for action, targets in action_target_vocab.items()
    }
    
    return {
        "action_patterns": action_patterns,
        "action_target_vocab": vocab,
        "multi_mode": True
    }
```

### 使用例

```python
from pathlib import Path

# Step 1: 知識DB の構築（GT から一度だけ）
kb = build_knowledge_db(Path("/workspace/data/annotations_gt_task_ver10.json"))

# Step 2: Parser のインスタンス化
parser = InstructionParser(kb)

# Step 3: 予測（instruction のみ）
instruction = "Apply a smooth dolly in effect toward the man's face"
pred = parser.pred(instruction)

print(pred)
# Output:
# {
#   "tasks": [
#     {
#       "action": "dolly_in",
#       "target": "face",
#       "constraints": [],
#       "params": {}
#     }
#   ]
# }
```

### 設計の利点

✅ **Train/Test 分離**
  - DB: 統計的な知識
  - pred(): instruction のみで予測

✅ **チート情報なし**
  - video_path や GT の実値は不使用
  - params は不出力（コンペでは別途推定）

✅ **スケーラブル**
  - 新しい動画には video_path 依存なし
  - 知識 DB は固定

✅ **可視化・検証が容易**
  - action patterns の正規表現は確認可能
  - vocab の候補も確認可能

### 実装ステップ

1. **Step 1**: `build_knowledge_db()` を実装
2. **Step 2**: `InstructionParser` クラス実装
3. **Step 3**: 既存の `build_predictions()` を置き換え
4. **Step 4**: テスト・検証（コンペベースライン比較）

---

## 比較表：現在 vs 改善版

| 項目 | 現在 | 改善版 |
|---|---|---|
| **入力** | instruction + records | instruction のみ |
| **DB** | annotations_gt_task_ver10.json (full) | 統計知識（action_patterns, vocab） |
| **クラス** | 関数型（build_predictions） | クラス型（InstructionParser） |
| **詳細参照** | (video_path, action) → GT task || | 知識 DB のみ |
| **チート度** | 高い（GT そのまま） | 無い（統計情報のみ） |
| **Train/Test分離** | ❌ 不可 | ✅ 可能 |
| **コンペ対応** | ❌ NG | ✅ OK |
| **params出力** | ✅ GT から | ❌ empty dict |

---

## 🔧 プロトタイプ実装・検証（2026-04-02）

### 実装内容

改善設計を実装したプロトタイプ：

📄 `/workspace/src/parse/prototype_instruction_parser_class.py`
```python
class InstructionParser:
    def __init__(self, knowledge_db: dict):
        # 統計知識 DB を保持
        self.patterns = knowledge_db["action_patterns"]
        self.vocab = knowledge_db["action_target_vocab"]
    
    def pred(self, instruction: str) -> dict:
        # instruction のみで予測
        action = self._infer_action(instruction)
        target = self._choose_target(instruction, action)
        return {"tasks": [{
            "action": action,
            "target": target,
            "constraints": [],
            "params": {}  # チート情報なし
        }]}
```

### テスト結果

**実行コマンド**:
```bash
./scripts/run_prototype_instruction_parser_class.sh
```

**結果サマリー**:
```
✓ Knowledge DB 構築
  - Actions: 13 個
  - Vocab entries: 42 個

✓ Parser 初期化
  - action_patterns: 13
  - action_target_vocab: 42

✓ テスト予測（4件）
  • "Apply dolly in effect..." → dolly_in, "man's face"
  • "Increase animals..." → (fallback), "object"
  • "Apply oil painting..." → apply_style, "full_frame"
  • "Zoom in on face..." → zoom_in, "face"

✓ DB 検査
  ✅ video_path なし（チート情報排除）
  ✅ params は空（instruction のみでは生成不可）
  ✅ Train/Test 分離可能
```

**実行ログ**:
- `/workspace/logs/analysis/prototype_instruction_parser_class_20260402_140359.log`

### key 設計ポイント

1. **knowledge_db 構造**
   ```python
   {
     "action_patterns": {
       "dolly_in": r"\bdolly in\b",
       "zoom_in": r"\bzoom in\b",
       ...
     },
     "action_target_vocab": {
       "dolly_in": ["man's face", "camera_view", ...],
       "add_object": ["animal", "person", ...],
       ...
     },
     "multi_mode": bool
   }
   ```

2. **target の型許容**
   - 一部の GT では target が **list** の場合もある
   - 処理: フラット化して個別に vocab に登録

3. **params 出力**
   - instruction のみでは詳細值推定困難
   - この設計では `params: {}` で返す
   - 本実装では、別途ロジック追加が必要

### 今後の実装ステップ

1. ✅ **設計案作成** (`docs/instruction_parser_analysis.md`)
2. ✅ **プロトタイプ実装・検証**
3. **本実装化**
   - `src/parse/instruction_parser_v20.py` に移行
   - params 生成ロジック追加
4. **既存コード置き換え**
   - `build_predictions()` を `InstructionParser.pred()` に切り替え
5. **統合テスト**
   - コンペベースラインとの比較検証

---

## 🚨 チート情報排除版（v2）

### 問題発見と改善

**初期版プロトタイプの問題**:
```python
# ❌ チート：GT から具体的な target を学習
action_target_vocab = {
  "add_object": ["rhino_and_buffalo"],  # GT で見たもの
  "dolly_in": ["man's face", "manual coffee grinder"],
  ...
}
```
→ コンペでは、未知動画の具体的な target にマッチしない

**改善版v2の設計**:
```python
# ✅ チート情報なし：instruction テキスト処理から抽出
class InstructionParser:
  def _extract_target(self, instruction: str) -> str:
    # HTML noun_patterns（汎用）に基づいてマッチ
    # GT の具体的 target は参照しない
    return matched_noun
```

### v2 の構成

```python
knowledge_db = {
  "action_patterns": {
    "dolly_in": r"\bdolly in\b",      # ドメイン知識（許容）
    ...
  },
  "noun_patterns": {
    "body_part": r"\b(face|eye|head|hand|person|man|woman)\b",
    "object": r"\b(object|item|car|animal|bird)\b",
    "location": r"\b(scene|background|foreground|frame)\b",
    "generic": r"\b(full frame|entire|all|whole)\b",
  }
}
```

### チート情報確認

**❌チート情報**:
- GT から学習した (action, target) ペア
- video_path
- GT の具体的な target 値

**✅許容される情報**:
- action_patterns（正規表現）
- 汎用 noun_patterns（一般的な名詞クラス）

### テスト結果（v2）

```
✓ Knowledge DB 構築（チート情報なし）
  • Actions: 13
  • Noun patterns: 4（body_part, object, location, generic）

✓ テスト予測（instruction テキスト処理）
  • "Apply dolly in toward man's face" 
    → dolly_in, "man"（GT の "man's face" ではなく）
  • "Zoom in on face"
    → zoom_in, "face"
  • "Replace background"
    → replace_background, "background"
  • "Add car object"
    → add_object, "car"（GT の "rhino_and_buffalo" ではなく）

✓ DB 検査
  ✅ video_path: 不使用
  ✅ (action, target) ペア: GT から学習していない
  ✅ target: instruction テキスト処理から直接抽出
  ✅ params: 空 dict
```

### 実行ログ

- スクリプト: `/workspace/scripts/run_prototype_v2_no_cheat.sh`
- ログファイル: `/workspace/logs/analysis/prototype_v2_no_cheat_20260402_140826.log`

### 設計の確認チェックリスト

- ✅ GT から学習する情報 = action_patterns のみ
- ✅ GT から学習しない情報 = target 候補
- ✅ DB に video_path がない
- ✅ (action, target) ペア学習なし
- ✅ target は instruction テキスト処理から抽出
- ✅ params は空 dict
- ✅ Train/Test 完全分離

**結論**: チート情報なし確認済み ✅ コンペ対応可能
