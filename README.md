# TTS 評測平台

同一段文字送進多個 TTS 引擎，比三件事：

- **速度** — 合成耗時、RTF、即時倍速、首段延遲，外加 p95 與穩定性
- **吐字** — ASR 字錯誤率 CER（客觀）
- **好不好聽** — 盲測投票（對決或排名），用 Elo 排名

模型全部跑在別台主機上，這個平台只負責發請求、計時、算分、存結果，
所以本機不用裝 torch，`pip install` 十幾秒就好。

引擎是**設定驅動**的。四包 gateway（cosyvoice2 / fun-cosyvoice3 / qwen3-tts /
voxcpm2）用的是同一份 OpenAI 相容 API，所以不需要四個轉接器 ——
一個轉接器、四份設定，全部寫在 `engines.json`。加第五顆模型不用寫 Python。

```
                     ┌──────────────┐  :18001  cosyvoice2   clone
                     │              │  :18002  fun-cosyvoice3 clone
   ┌──────────┐      │   四包       │  :18003  qwen3-tts    clone
   │  評測平台  │ ───▶ │   gateway   │  :18004  voxcpm2      clone/design/default
   │ (這個專案) │      │ OpenAI 相容  │
   └──────────┘      └──────────────┘
        │
        └──────────▶ Whisper ASR（把合成結果轉回文字算 CER）
```

---

## 一、快速開始

```bash
# 1) 安裝（很輕，只有 FastAPI / httpx / soundfile）
python -m venv .venv-console && source .venv-console/bin/activate
pip install -r requirements-console.txt

# 2) 引擎清單
cp engines.example.json engines.json     # 四包 gateway 的 key / port / 特性
cp .env.example .env
#   TTS_HOST=192.168.1.223               ← 四包 gateway 所在的主機
#   ASR_BASE_URL=http://.../             ← 算 CER 用的 Whisper

# 3) 啟動
./run.sh
```

啟動時會把展開後的位址全部印出來 —— 主機填錯是最常見的問題，
所以讓它在第一行就看得見，而不是等第一次評測失敗才發現。

開 <http://127.0.0.1:8080>。六個分頁：**競技場 / 排行榜 / 批次 / 紀錄 / 音色 / 引擎**。

### 第一次使用的順序

1. **引擎** → 按「全部暖機」。冷啟動載模型要 30–90 秒，
   不先做這件事，第一筆量到的是「載模型 + 合成」，跟其他引擎完全不能比。
2. **音色** → 上傳一段 2–30 秒的乾淨人聲，填逐字稿（或按自動辨識）。
   上傳完會自動推送到每一顆能克隆的引擎。
3. **批次** → 選「公開」模式跑完整題庫，先拿到客觀普查（速度 / CER / 分類戰報）。
4. **競技場** → 主觀盲測，累積 Elo。
5. **排行榜** → 先按「量 ASR 誤差底線」，再開始解讀數字。

---

## 二、四包 gateway 的能力差異，以及它怎麼影響比較

| 包 | port | 支援的 mode | 音色可比？ | `speed` 有效？ | 這輪 |
|---|---|---|---|---|---|
| fun-cosyvoice3 | 18002 | `clone` | ✅ | ✅ | ✅ 測 |
| qwen3-tts | 18003 | `clone` | ✅ | ❌ 會忽略 | ✅ 測 |
| cosyvoice2 | 18001 | `clone` | ✅ | ✅ | ⏸ `enabled: false` |
| voxcpm2 | 18004 | `clone` / `design` / `default` | ✅ | ❌ 會忽略 | ⏸ `enabled: false` |

四包同時上模型太久，所以這輪先只跑 **fun-cosyvoice3 + qwen3-tts**（有二／三代的取三代）。
另外兩包在 `engines.json` 裡留著，只是加了 `"enabled": false` —— 之後要補測，
把那一行拿掉，按 UI 上的「重載 engines.json」就回來了，設定不用重寫。

平台**不寫死**上面這張表。`modes` / `presets` / `sample_rate` / `loaded`
全部是從各 gateway 的 `GET /v1/models` 探測來的 —— gateway 自己就是這樣設計的
（`ENGINE_MODES` 環境變數決定路由）。qwen3-tts 從 CustomVoice 換成 Base
checkpoint（`ENGINE_MODES=clone`）時就驗證過這件事：平台這邊一行都不用改，
探測到 `clone` 就自動改用音色庫的參考音，音色維度也自動變成可比。

注意 Base checkpoint **沒有**內建 speaker（Vivian / Ethan 那組是 CustomVoice
專屬）。所以 qwen3-tts 一定要先推送音色，音色庫是空的時候合成一律 400。

`engines.json` 裡只放 API 探測不到的東西：顯示名稱、`speed` 有沒有效、
語言、preset 要用哪個 speaker。

### 音色為什麼要「推送」

四包 gateway 的音色庫（`work/voices/`）**彼此不共用**。同一段參考音沒有推到每一包，
每顆引擎就會拿自己手上剛好有的音色去合成 —— 那個「音色相似度」是在比不同的東西。

所以本機音色庫是唯一的真實來源，平台負責同步：

```
storage/voices/<id>/prompt.wav
    ├─ POST :18001/v1/voices  →  voice_a1b2…
    ├─ POST :18002/v1/voices  →  voice_c3d4…
    └─ POST :18004/v1/voices  →  voice_e5f6…
```

對照表在 `storage/voice_map.json`，記了參考音與逐字稿的雜湊。
**改了逐字稿或換了參考音，遠端那份會被判定成 `stale`，而且不會被拿來評測** ——
拿舊的參考音去比新的，比出來的東西沒有意義。下次評測前會自動重推。

### 幾個會偷換受測條件的坑，平台的處理方式

| 坑 | 為什麼會出事 | 平台怎麼做 |
|---|---|---|
| `instructions` | CosyVoice 收到就會走 `inference_instruct2`，**不再走 zero-shot**，音色相似度跟不給時不一樣 | 預設不送。要送得在 `engines.json` 開 `allow_instructions` |
| `speed` | 只有 CosyVoice 兩顆真的會用，另外兩顆直接忽略 → 調了語速，四個模型有兩個沒反應 | 每顆標 `speed_mode`；`local` 的改由本機變速不變調，**不計入合成耗時** |
| 沒填逐字稿 | CosyVoice 退到 cross-lingual、VoxCPM2 少掉 ultimate cloning，相似度都明顯下降 | 缺逐字稿的音色直接擋下來，不讓它進評測 |
| 冷啟動 | 第一次合成要載 30–90 秒模型 | 暖機打 `POST /v1/warmup`，並讀 `/v1/models` 的 `loaded` 確認 |
| 能力快取 | gateway 對引擎 `/health` 有 300 秒快取，冷啟動時 `presets` 是空的還會被快取住 | 暖機會清 gateway 那層，平台這層也跟著清 |

---

## 三、三種評測模式

| 模式 | 做什麼 | 什麼時候用 |
|---|---|---|
| **公開** | 跑全部，直接列數據，不盲測 | 換模型後的第一輪普查、批次跑測 |
| **對決** | 跑全部，但只**盲測其中兩個**讓你二選一 | 累積 Elo 的主力模式 |
| **排名** | 跑全部、全部盲測，依序點出 1–4 名 | 想一次拿到最多主觀資訊 |

### 對決為什麼「跑全部、只比兩個」

速度與 CER 是客觀的，多跑一顆就多一筆客觀資料，反正都已經在等了。
但主觀投票是二選一才準：一次聽四段再排序很累，也容易被順序影響。
所以客觀資料吃滿、主觀負擔壓到最低。

### 要配哪兩個？不是隨機

純隨機在四個模型（6 種組合）時會讓某些組合遲遲配不到，對戰矩陣一直有空格，
Elo 也會因為對手覆蓋不均而失真。平台會**優先配目前交手次數最少的那一對**，
讓矩陣均勻長滿。

### 排名怎麼變成 Elo

四方排名（`C > A > D > B`）會拆成 6 組成對結果，每兩個之間都有大小關係。
但這 6 組**不是互相獨立的** —— 它們來自同一次聆聽、同一個判斷。當成 6 場獨立
對戰灌進 Elo 會嚴重高估這一筆的份量，所以排名來源的每一組 K 值除以 `(n-1)`。

「兩個都不行」刻意**不產生**任何成對結果：它說的是「兩個都不及格」，
並沒有說誰比較好，記成平手會污染排名。

---

## 四、盲測是真的盲測

投票前 API 只回格子代號與音檔網址，其他一律不給：

- 引擎名稱藏起來（基本）
- **秒數也藏起來** —— 兩個引擎速度差好幾倍，看到「1.6 秒 vs 11.4 秒」就等於知道答案
- **參賽名單也藏起來** —— 知道候選是誰就會帶著預期去聽
- 可比性說明裡有引擎名稱，未揭曉時只留布林值
- 音檔走 `/api/trials/{id}/audio/{格子}`，下載檔名也只有格子代號

`scripts/smoke_test.py` 會逐項驗證這幾件事。

---

## 五、指標怎麼讀

### 主排名用 Elo，不是勝率

四個模型兩兩對戰時勝率會騙人：常抽到弱對手的模型勝率虛高。
Elo 會把「贏了誰」算進去 —— 贏強的加比較多分，輸給弱的扣比較多分。

排行榜上每個引擎的 Elo 都帶一條**90% 信賴區間**（bootstrap 出來的）。
**區間重疊就代表以目前的樣本分不出高下** —— 樣本少的時候 Elo 點估計幾乎沒有意義，
這條區間就是在擋過度解讀。

### 速度：看中位數，也要看 p95 跟 CV

平均值會藏住尾巴。一個平均 2 秒但偶爾爆到 15 秒的模型，
跟穩定 3 秒的模型，看平均前者贏，但實際用起來後者好得多。

- **p95** — 最糟的那 5% 有多糟
- **穩定性 CV** — 標準差 ÷ 平均，跟單位無關所以跨引擎可比。
  把「重複次數」設成 3 或 5，這欄才有東西

### CER 是相對指標，不是絕對品質分數

Whisper 自己就會聽錯 —— 拿真人原始錄音去辨識也不會是 0。
所以排行榜有「量 ASR 誤差底線」：拿音色庫裡的真人錄音跑一次，得到這個 ASR
的誤差下限。引擎的 CER 接近這個數字，就代表吐字已經沒問題了。

平台會自動比對：**兩個引擎的 CER 差距如果小於 ASR 誤差底線的 1/4，
就會在「判讀提醒」裡說這個差距不足以下結論。**

還有一件事平台**沒有**幫你處理：`normalize_for_cer()` 只去標點、空白與全半形，
不做簡繁轉換、也不統一數字寫法。所以 CER 裡混了兩種與引擎品質無關的誤差 ——
Whisper 有時吐簡體字（整段都算錯）、輸入寫「兩萬三千四百五十六」而 ASR 寫「23456」。
這個雜訊對不同引擎的影響幅度不一樣，足以翻轉分類名次。要下結論之前，
用 [`scripts/bench_report/`](scripts/bench_report/) 另外算一版**寬鬆 CER**，兩版一起看。

### 分類戰報

「哪個模型最好」通常沒有單一答案，而是「數字誰強、破音字誰強」。
題庫按痛點分類（短句 / 播報 / 數字 / 中英夾雜 / 破音字 / 語氣停頓 / 罕用字 / 長段落），
排行榜會拆開來看。這張表才是換模型時真正在看的東西。

### 音訊客觀指標（不需要 ASR 也不需要人耳）

| 指標 | 抓什麼 |
|---|---|
| **頻寬比** | 實際內容頻寬 ÷ 宣稱取樣率的奈奎斯特頻率。明顯小於 1 = **宣稱 48kHz，其實是低取樣率模型升採樣上來的** |
| 靜音比例 | 引擎在段落間塞了多少空白 —— 太高的話音檔長度灌水，RTF 要打折看 |
| 開頭 / 結尾靜音 | 即時應用的體感延遲 |
| 削波比例 | 輸出爆音，音量沒收好 |
| 語速（字/有聲秒） | 太快可能吃字，太慢會拖 |

這些都在音量正規化**之前**算 —— 正規化會把每個引擎的峰值拉到同一個數字，
peak / RMS / 削波全部會失去比較意義。

---

## 六、批次跑測

一句一句手動送，四個模型跑十句就要點四十次，而且中途很容易改到設定
（換音色、改語速），最後拿到的是一堆條件不一致的資料。

批次把「同一組設定、同一批文字、同一批引擎」綁成一次任務，跑完存成一份報表
（`storage/reports/batch-*.json`）。每一句仍然是一筆正常的 trial，會算進排行榜，
只是多帶一個 `batch_id`。

- **公開模式** — 只收客觀數據，跑最快
- **對決 / 排名模式** — 每一句都留成待投票的盲測，跑完到競技場慢慢聽慢慢投

批次**依序**跑（不並行），理由跟單筆評測一樣：引擎共用同一台 GPU。
十句 × 四個引擎會跑好幾分鐘。

### 跑完之後：離線分析

報表只有那一批的摘要。要做跨引擎、跨批次的比較（寬鬆 CER、成對顯著性檢定、
把不同批次的同一句對起來），用 [`scripts/bench_report/`](scripts/bench_report/)：

```bash
make install-bench          # 只多一個 zhconv
make bench-report ENGINES="fun-cosyvoice3=batch-A qwen3-tts=batch-B f5=batch-B"
```

引擎不是同一批跑的時候它會警告 —— 跨場次的速度比較混著系統性誤差，
不能當成同場結果讀。

---

## 七、加一個新引擎

**如果它也是這套 gateway 或任何 OpenAI 相容端點** —— 不用寫程式：

```json
{
  "key": "my-model",
  "label": "我的模型",
  "base_url": "http://${TTS_HOST}:18005",
  "speed_mode": "local",
  "preset_voice": "",
  "language": "",
  "required_fields": [],
  "notes": "備註會顯示在引擎分頁上"
}
```

存進 `engines.json`，按 UI 上的「重載 engines.json」就生效，不用重啟。
能力會自己從 `/v1/models` 探測。

`required_fields` 是給「欄位不能省略」的 API 用的。多數 OpenAI 相容端點空欄位不送
就好；列在這裡的欄位會在送出前先檢查，訊息直接告訴你要補哪一個，而不是等遠端
回一句看不出要填哪裡的 validation error。`qwen3-tts` 這個 key 已經內建
`["language"]`（它的 `language` 不指定會退到 `Auto`，克隆路徑的品質比較不穩），
不用自己寫。

**介面完全不一樣的話**，在 `app/engines/` 開一個檔案：

```python
class MyEngine(TTSAdapter):
    key = "myengine"
    label = "My Engine"
    needs_voice = False

    @property
    def enabled(self):
        return bool(config.MY_BASE_URL)

    def request_segment(self, segment, opts):
        r = httpx.post(url, json={"text": segment})
        return decode_audio_bytes(r.content)     # -> (float32 波形, 取樣率)
```

然後在 `app/engines/__init__.py` 的 `_LEGACY_BUILTIN` 加一行。
切段、計時、拼接、音量正規化、音訊指標都由基底類別統一處理 ——
這不是為了少寫幾行，而是**評測的正確性要求**：兩個引擎如果用不同的切法或
不同的計時起點，測出來的秒數就不能拿來比。

---

## 八、專案結構

```
engines.json      引擎清單 —— 加模型改這裡就好（base_url 支援 ${TTS_HOST}）
app/
  main.py         FastAPI 路由（不認識任何具體引擎，只透過登錄表）
  bench.py        評測核心：跑一次 trial、盲測配對、遮蔽、投票
  stats.py        Elo、bootstrap 信賴區間、對戰矩陣、分佈量化
  report.py       排行榜、分類戰報、判讀提醒、CSV 匯出
  batch.py        批次跑測與報表
  voice_sync.py   把本機音色推送到各引擎，記對照表與過期偵測
  audio_metrics.py 音訊客觀指標（頻寬、靜音、削波、語速）
  engines/
    __init__.py   登錄表 —— 從 engines.json 動態建出來
    base.py       TTSAdapter 基底：共用切段／計時／拼接／指標流程
    gateway.py    四包 gateway 的轉接器（一個類別、多個實例）
    f5_tts.py     舊 F5-TTS 轉接器（介面不同，獨立一個）
  asr.py          遠端 Whisper 用戶端 + CER 評分
  voices.py       本機音色庫（每個音色一個資料夾：prompt.wav + meta.json）
  jobs.py         單執行緒任務佇列（依序執行是評測正確性的要求）
  text_utils.py   注音安全的切段 + CER 正規化與編輯距離
  audio_utils.py  讀寫 / 重採樣 / 拼接 / 正規化
  static/index.html  前端（單檔，無建置流程）
storage/
  voices/<id>/    prompt.wav（16k 單聲道）+ meta.json
  voice_map.json  本機音色 → 各引擎遠端 voice id 的對照表
  outputs/        評測產生的音檔
  trials.json     評測紀錄（含投票結果）
  reports/        批次報表
scripts/
  smoke_test.py         不需要任何遠端服務的完整流程測試（--repeat 可連跑找不穩定）
  bench_report/         批次跑完之後的離線分析（不是平台的一部分，單向依賴 app）
    lenient_cer.py      寬鬆 CER：把簡繁與數字書寫形式的雜訊扣掉
    significance.py     成對 Wilcoxon／符號檢定／中位數信賴區間
    build.py            把多顆引擎的逐句數據對齊，組成比較報告的資料檔
requirements-bench.txt  離線分析要的套件（平台與 CI 都不需要）
Makefile                make test / test-ci / test-flaky / bench-report
.github/workflows/smoke.yml   CI：每次 PR 跑一次，每天掃一次 flaky
```

---

## 九、API

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/api/status` | 引擎能力、ASR 狀態、**哪些維度可比** |
| GET | `/api/health/{engine}?deep=true` | 單一引擎健康檢查；`deep` 會實際合成一句 |
| POST | `/api/engines/reload` | 重讀 `engines.json`，不用重啟 |
| POST | `/api/warmup` | 把模型載進 GPU（可帶 `engines`） |
| GET | `/api/presets` | 內建測試句與分類 |
| POST | `/api/trials` | 送出評測 → `job_id`（`text`, `engines`, `mode`, `repeats`, `voice_id`, `score`, `speed`） |
| GET | `/api/trials/pending` | 還沒投票的盲測 |
| GET | `/api/trials/{id}/audio/{slot}` | 盲測音檔（不洩漏引擎） |
| POST | `/api/trials/{id}/vote` | 投票並揭曉（`choice`: 格子／`tie`／`bad`，或 `ranking`: 格子陣列） |
| POST | `/api/trials/{id}/reveal` | 不投票直接揭曉（不列入排名） |
| POST | `/api/voices/{id}/push` | 把參考音推送到各引擎 |
| GET | `/api/voices/{id}/sync` | 這個音色在各引擎上的推送狀態 |
| POST | `/api/batches` | 批次跑測 → `job_id` |
| GET | `/api/batches/{id}` | 批次報表 |
| GET | `/api/leaderboard` | Elo、速度、CER、分類戰報、對戰矩陣、判讀提醒 |
| GET | `/api/export.csv` | 一列一個（評測 × 引擎）的攤平資料 |
| POST | `/api/asr-baseline` | 量 ASR 誤差底線 |
| GET | `/api/jobs/{id}` | 輪詢任務進度 |

評測是**非同步**的：`POST /api/trials` 立刻回 `job_id`，輪詢 `GET /api/jobs/{id}`，
`status` 走 `queued` → `running` → `done`，`progress` 是 0–1。

```bash
JOB=$(curl -s localhost:8080/api/trials -H 'Content-Type: application/json' \
  -d '{"text":"今天天氣真好。","engines":["cosyvoice2","voxcpm2"],"mode":"duel"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
curl -s localhost:8080/api/jobs/$JOB | python3 -m json.tool
```

設了 `API_KEY` 的話，每個 `/api` 請求都要帶 `X-API-Key`。

---

## 十、疑難排解

**引擎顯示「待暖機」，第一筆特別慢**
模型還沒載進 GPU（`/v1/models` 的 `loaded: false`）。按「全部暖機」，
冷啟動要 30–90 秒。這是刻意分開的：載模型的時間不該算進合成耗時。

**能力探測看到的是冷啟動時的舊狀態**
`modes` / `presets` / `loaded` 是**模型載入時才填的**，而 gateway 對引擎能力有
300 秒快取，冷啟動後太早打會看到「還沒 ready」、而且會被快取住五分鐘。
按「暖機」，它會順手清掉那層快取。

**一直回 503**
gateway 連不到引擎，通常還在載模型。平台會自動重試（3 / 8 / 15 秒退避），
還是不行就確認那個容器有沒有起來。

**音色標成「推送不足兩顆，音色不可比」**
克隆比較至少要有兩顆引擎拿到同一段參考音。到「音色」按「推送到各引擎」，
看每顆的狀態標籤。四包現在都吃克隆，沒有 `unsupported` 是正常的。

**音色標成「已過期」**
本機的參考音或逐字稿改過，遠端那份是舊的。平台會**拒絕拿它去評測**
（不然就是拿舊參考音比新的）。下次評測前會自動重推，或手動按「推送到各引擎」。

**排行榜說「Elo 信賴區間重疊」**
就是字面意思：以目前的樣本分不出高下。多投幾筆，區間會收窄。
不要因為點估計差 20 分就下結論。

**某個引擎的「頻寬比」特別低**
它宣稱的取樣率高，但實際內容的頻寬遠低於奈奎斯特頻率 ——
通常是低取樣率模型升採樣上來的。盲測時高取樣率天生佔便宜，
判斷音質差異時要把這點算進去（排行榜的判讀提醒也會講）。

**速度數字忽大忽小**
先按「暖機」。另外評測佇列是單執行緒的，同時只會有一筆在跑 ——
這是刻意的，引擎在同一台機器上，並行測出來的秒數不可比。
想量穩定性就把「重複次數」設成 3 或 5，看 CV 那一欄。

**改了 engines.json 沒生效**
按「引擎」分頁的「重載 engines.json」。Docker 的話那個檔案是掛進去的，
改宿主機上的檔案就好，不用重建映像。

---

## 十一、開發

```bash
make test                    # 跑一次完整流程測試（等同 python scripts/smoke_test.py）
make test-ci                 # 只印失敗，另外產出 JUnit XML 與 JSON 報告
make test-flaky N=50 J=4     # 連跑 50 次（同時 4 個），抓時好時壞的項目
```

會把每個轉接器的 HTTP 呼叫與 ASR 換成假的，驗證：四方評測、盲測遮蔽
（引擎名稱／秒數／參賽名單）、排名拆成成對結果、Elo 排序、對決配對覆蓋、
音色推送的過期偵測、升採樣偵測、重複跑的輪替順序、批次報表、CSV 匯出、路徑安全。
最後再用一個真的 HTTP stub 檢查 gateway 轉接器送出去的 payload。
**完全不連遠端，所以 CI 上不需要 GPU、不需要 `.env`、也不需要 `engines.json`。**

### 自動化模式

```
--quiet            只印失敗的項目與總結（通過的不刷版面）
--json 檔案        逐項結果寫成 JSON（填 - 印到 stdout）
--junit 檔案       JUnit XML，GitHub Actions / GitLab / Jenkins 都讀得懂
--repeat N         連跑 N 次，統計逐項成功率
--jobs N           --repeat 時同時跑幾個
--no-color         不要 ANSI 色碼（偵測到 CI 環境變數時自動關）
--keep-tmp         通過時也保留暫存資料夾（預設只在失敗時留著）
```

全部通過回傳 0，有任何一項失敗或中途爆掉回傳非 0。

`--repeat` 是拿來找**不穩定的測試**的：每次都開獨立子行程（同行程重跑沒有意義 ——
暫存資料夾、被換掉的假物件、config 讀進來的設定都已經被上一次污染了），
跑完把項目分成三類：

- **不穩定** —— 20 次裡失敗 2 次，這種才是要修的，跑一次看不出來
- **每次都失敗** —— 真的壞了
- **沒跑完** —— 有幾次在它之前就爆掉了（分母會標出來）

這條流程對時序特別敏感：評測佇列是單執行緒的、`elapsed_spread` 在量耗時分佈、
對決配對靠交手次數挑組合，任何一個地方混進非決定性都會變成偶爾失敗的測試。

### CI

`.github/workflows/smoke.yml`：

- push 到 main 與每個 PR → 跑一次，報告存成 artifact
- 每天 02:00（台北）與手動觸發 → 連跑 20 次掃 flaky

---

## 十二、授權

本平台程式碼為專案自有。各 TTS 引擎跑在遠端服務上，授權依各自專案。
