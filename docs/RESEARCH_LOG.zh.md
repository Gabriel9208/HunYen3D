# 研究日誌（中文版）

在單張 RTX 5080 上訓練，使用約 45k 個 ShapeNet watertight meshes (3DShape2VecSet 資料集)。
試圖重現並改進 **Hunyuan3D-2 ShapeVAE** 的模組。

> **註記(2026-07-25)。** 先前所有 45k 泛化實驗都用了錯誤的 train/val 切分(跨類別、train 與 val
> 類別完全不重疊,是 zero-shot 而非 in-distribution),數字不可信,連同其對應的結論、圖表已全數移除。
> 唯一保留的是與切分無關的 Stage-1 單一 mesh overfit,以及各模組的方法/設計描述。以下結果一律改用
> 論文官方的 in-distribution 三分割(train/val/test)重新訓練與評估,逐一補回。

## 前言 —— 這個專案的基本設定
本研究是基於HunYuan3D 2.0 的開源專案以及技術報告，因此基礎模型架構接承襲於它。

### 重建部份： VAE 的 encoder、decoder 及相關模組。
- **任務與輸入輸出**：給定一個 mesh 的表面點,encoder 把它映射成一個 latent set;decoder 再針對任意 3D query 點預測其 **SDF**(有號距離:<0 在內部、>0 在外部、0 在表面上)。而 Occupancy / IoU 的數字是由 SDF 的正負號判斷(內部 = SDF<0)。
- **架構**：VecSet VAE:encoder → **一組 N 個 latent tokens,每個寬度 64**(`num_latents` N ∈ {1024, 3072};`latent_dim` = 64)→ KL → decoder → 每個 query 的 SDF。 Mesh 由預測出的 SDF 場經 **marching cubes(MC)** 抽出。
- **目標函數**： **SDF MSE** 以及 **KL loss**。
- **指標(metrics)**： VIoU, SIoU, F-score, NC, Chamfer Distance

## Stage-1 —— 容量測試: overfit 一個 mesh

*(此段只 overfit 單一 mesh、沒有 train/val 切分,不受切分錯誤影響,數字仍可信,故保留。)*

**目的。** 先證明架構*能*記住單一形狀(排除「容量不足」)。

**受控條件:**

| 參數 | 值 | 為何固定 |
|---|---|---|
| train mesh | 1 個固定(`sorted(paths)[0]` = 一架飛機),`max_meshes=1` | 學習同一個資料 |
| model | cell A:2048 latents,enc4/dec8,width 1024,16 heads,latent_dim 64 | 最小格 = 單調容量下限 |
| `kl_weight` | 0 | 純容量測試先忽略 regularization term |
| `deterministic` | true(z=μ) | 只取μ，σ多少都可以  → 即使 kl=0 也沒 σ-explosion 問題 |
| lr | 1e-4 | 1e-3 在 overfit 會發散;1e-4 穩定下降 |
| eval | μ-path(在 z=μ 訓練),MC **res-256** | 256 暴露 res-128 會抹平的表面粗糙 |
| PASS 門檻 | train RMS < 0.02 | = near-band 寬度 |

**變動的參數:**

| 參數 | 掃描 | 發現 |
|---|---|---|
| `fixed_seed` | 0(凍結)→ null(每個 epoch 重採樣) | **若是固定seed，模型就永遠只會看到相同的一組 query 點被監督，就算模型把所有query點記起來，點與點之間也沒有被監督。** |
| LR schedule | constant → cosine decay(1e-4→1e-6) | Constant-LR + 重採樣把 loss 卡在 SGD 噪聲球(目標每 epoch 都在動)。Cosine decay 把它壓定:loss 2.3e-4 → 1.4e-5,救回重採樣單獨造成的保真度損失。 |
| epochs | 3000 → 6000 → 12000 | 更多 epochs 只在 *seed 解凍且 LR decay 之後*才有幫助;更多凍結-seed epochs 只是把那 2048 個點 overfit 得更死。 |

**結果(同一 ckpt 血統,eval res-256,μ-path):**

| 版本 | NC(平滑) | Chamfer | F-score@.02 | V-IoU | S-IoU | RMS | train loss |
|---|---|---|---|---|---|---|---|
| frozen seed, const-LR, 3k ep | 0.712 | 0.0124 | 0.993 | 87.0% | 83.6% | 0.0036 | ~0 |
| unfrozen, const-LR, 6k ep | 0.860 | 0.0247 | 0.947 | 64.0% | 58.1% | 0.0119 | 2.3e-4 |
| **unfrozen, cosine-LR, 12k ep** | **0.943** | **0.0087** | **1.000** | **92.3%** | **90.2%** | 0.0035 | **1.4e-5** |

**判定:PASS。** 架構能把單一 mesh 記到 F-score 1.00 / NC 0.94 / Chamfer 0.009 / S-IoU 90%,渲染出來肉眼也對得上。

## Stage-2 —— Baseline VAE 20 Epoch

`vae_baseline_1024_l8_16`(1024 tokens、enc8/dec16、plain MSE、kl1e-3、cosine 3.5e-5→1e-6、20 ep;
論文官方 in-distribution split、從頭訓練)。這是非 anchor 的參考 baseline。

**後驗 μ-broken。** 這個 baseline 的 μ-spread 很小(0.14–0.20)、σ_rms 幾乎貼著 prior(≈0.97),使
decode(μ) 落在樣本殼之外 → μ-path 評估直接崩壞。由於下游 diffusion 擬合的是 *sample* 而非 μ(領域
標準也是解碼樣本),因此以 **sample-avg** 為準。

| eval | V-IoU | S-IoU | RMS | σ_rms | μ-spread |
|---|---|---|---|---|---|
| μ-path (256) | 9.32% | 7.52% | 0.216 | 0.969 | 0.202 |
| **sample-avg K=32 (64 shapes)** | **51.03%** | **44.36%** | 0.0203 | 0.972 | 0.144 |

*警語:sample-avg 目前只跑了 64 shapes(256×32 會 timeout),且尚無 mesh 指標(Chamfer/F/NC)。
待補跑完整的 256 SDF + 16 mesh,以與 anchor 版對齊、可比。*

## Anchor VAE —— 把每個 token 的表面 anchor 當 decoder conditioning

### 背景
騰訊官方針對 HunYuan3D 2.0 中的 shape generation 分支提出的改進版本 LATTICE 中，提出了兩階段生成的方法。 因為HunYuan3D 2.0生成的模型細節不夠精細，於是他們將其voxelize之後提取與模型表面的active voxel並作為第二個Hunyuan3D 2.5版本生成模型的輸入，以此方式引導非結構化的VecSet生成，模擬了基於2D網格的圖像生成方法的成功。

### 我的方法
與其跑兩次完整生成管線兩次，不如跑一次就好。我保留了 HunYuan3D 2.0 的原始架構（VAE + DiT），在其之上我設計了一個 Anchor 模組，試圖從 latent 提取絕對座標，作為參考點將資訊加入到 decoder 中，讓生成模型和 decoder 在生成時直接參考物體的絕對座標。以此達成將 where 跟 what 的解隅，與論文提出的目的相同，但以不同方式達成。

### 問題
**Q1. Latent 是由 Diffusion 生成的，應該無法利用 Anchor 資訊，畢竟 Anchor 就是以 Latent 為輸入？**

A1. 有研究發現2D影像以及影片的diffusion生成模型會在早期階段生成影像大致雛型，後期才會開始雕細節。而經過測試發現這也成立於3D領域中。因此我將在diffusion生成的後半所有time step中抽取其中間產物z，並計算它當時預測的z^，以此作為anchor model的輸入。

**Q2. Anchor 能有效的提取座標嗎？**

A2. 可以，前提示它跟 VAE 同時訓練，並非先單獨訓練完 VAE 後再訓練 Anchor。 另外，訓練時 Anchor 的 gradient 必須傳回 VAE encoder，因為經實驗測試，若兩者detach教會非常難訓練（moving target）。但尚未證實是否可以直接在已經訓練好的 VAE encoder 上訓練 Anchor，這將會是之後研究的實驗。

### 設計

**Anchor VAE (`AnchorVAE`)：**
1. 設計一個 **`Anchor` module**對取樣的 z 做 6 層 self-attn 經過 linear layer `proj_anchor` 後獲得每個 token 的原始座標 `(B, num_latents, 3)`
2. 利用 `FourierEmbedder`(和 SDF query 點同一個 PE)編碼anchor座標;
3. 一個 **`AnchorDecoder`**,在 SDF-query 的 cross-attention 前面多一個 cross-attention(`anchor_cross_attention`，anchor 作為其KV，latent 作為Query)，讓 latent 可以根據 anchor 資訊，重建成更精細的 3D 模型。

**監督**：
1. 將 anchors 拉向 **FPS query 點的原始 xyz**,用 **chamfer-L1(主要監督, 可微 `torch.cdist` 雙向 NN,weight 1.0)+ index-aligned MSE(輔,weight 0.1)**。
2. Config 旋鈕 `anchor_chamfer_weight` / `anchor_mse_weight` 預設 0 → base VAE 與所有先前 run 不變。
3. 原始 `anchor_cd` / `anchor_mse` 記到 wandb(train + val)。

### 實驗數據

`anchor_baseline_1024_l8_16`(AnchorVAE、與上方 baseline 同配方:kl1e-3、lr 3.5e-5 cosine、20 ep、
新 split)。與 plain baseline 不同,anchor 監督把 μ 撐開,**後驗健康**(μ-spread 0.42、σ_rms 0.73),
不過 sample-avg 仍明顯高於 μ-path。

| eval | V-IoU | S-IoU | RMS | σ_rms | μ-spread |
|---|---|---|---|---|---|
| μ-path (256) | 48.46% | 39.81% | 0.0189 | 0.733 | 0.419 |
| **sample-avg K=32 (256 shapes)** | **66.42%** | **59.87%** | 0.0135 | 0.733 | 0.419 |

*警語:mesh 指標(Chamfer/F/NC)這輪被中止未完成,待補。baseline 目前是 64-shape、anchor 是
256-shape,shape 數不對等 → 兩者都要一次同協定(256 SDF + 16 mesh、sample-avg)的完整 eval 才能
正式對照。舊 split 上「anchor 勝 baseline」的關係尚待在新 split 重新驗證。*

### Converge run —— DONE + eval(2026-07-28)

`anchor_converge_1024_l8_16`(kl1e-3 anchor 配方、constant LR 3.5e-5 訓到 plateau,再從 best.pt 以
**lr÷10 = 3.5e-6** 續訓沉定;新 in-distribution split、從頭)。以 **sample-avg**(K=16)評估(此後驗
μ-path 會低估)。這次也**加了 volume-IoU** —— 只在均勻 volume 點上算 occupancy IoU,對齊論文協定
(OccNet / 3DShape2VecSet / Hunyuan 的 IoU 都是這樣取點),有別於我們預設那個 80% 近表面、偏難的 V-IoU。

| eval | V-IoU (all) | V-IoU (uniform, 論文協定) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | RMS | μ-spread |
|---|---|---|---|---|---|---|---|---|
| SDF 256 shapes | 76.23% | **87.34%** | 71.04% | — | — | — | 0.0107 | 0.327 |
| mesh 16 shapes res-128 | 81.23% | **88.79%** | 77.27% | **0.0117** | **0.979** | **0.940** | 0.0103 | 0.265 |

*(兩列都 sample-avg K=16;256-shape 的 SDF 指標較有代表性,16-shape 是 mesh-viz 慣例、才有 Chamfer/F/NC,16/16 全抽到面。)*

→ **表面保真度基本上 SOTA 級**:F-score **0.979**、Chamfer **0.0117**、NC **0.94**,跟 3DShape2VecSet 報的
F-score(~0.97)同級甚至更好。**volume-IoU ~88%** 落在論文區間;唯一偏低的是 **S-IoU(近表面符號)~71-77%**,
那是 SDF-MSE 表面梯度消失的結構弱點,與表面品質是兩回事。**關鍵教訓:先前「卡 0.72」是量錯了指標** —— 用
80% 近表面點算 IoU、而非論文的均勻 volume 點;換成 volume-IoU 立刻回到論文區間。渲染圖(GT↔recon,16/16)
在 `results/converge_meshsample_viz/`。

**生成探針(latent 健康度)。** `scripts/sample_shapes.py` 抽 latent 直接 decode:posterior(z~N(μ,σ))
與 aggpost(聚合後驗)都能 decode 出完整形狀(~12k+ verts),但純 prior(z~N(0,I))退化成碎片(~1.2k
verts)—— decoder 健康、但 latent 沒對齊 N(0,1)。**這正是留給 DiT 的 gap。**

## 指標

Chamfer-L1 + V-IoU + S-IoU + Normal-Consistency + F-score@0.02,全在 `src/metrics/`,mesh 為空/無法計算時每個指標有 N/A fallback。各指標的範圍 / 方向 / 尺度錨點:

| 指標 | 量測什麼 | 範圍 | 好 | 錨點 |
|---|---|---|---|---|
| **V-IoU** | 對**所有** query 點的 occupancy IoU(全域內外,occ = sign(SDF));時間軸裡光寫 **"IoU"** 指的就是 V-IoU | 0–1 | 越高 | overfit ~0.92 |
| **S-IoU** | V-IoU 限制在 **near band** `\|gt\|<0.02`(表面區) | 0–1 | 越高 | overfit ~0.90 |
| **Chamfer-L1** | 平均最近鄰表面距離(正規化座標 ~[−1,1]) | ≥0 | 越低 | overfit ~0.009 |
| **NC** | normal consistency = 配對表面法線的平均 \|cos∠\| | 0–1 | 越高 | overfit ~0.94 |
| **F-score@0.02** | τ=0.02 內表面點的 precision·recall | 0–1 | 越高 | overfit ~1.0 |

- **V-IoU 有兩種取點:** 預設 `V-IoU (all)` 是對整個 query bank(80% 近表面,偏難)算;`V-IoU (uniform)` 只用均勻 volume 點,**對齊論文(OccNet / 3DShape2VecSet / Hunyuan)的 volume-IoU 協定** —— 要跟論文比就看這個(evaluate.py 兩個都印)。80% 近表面那版會系統性地低估,別拿去跟論文對照。
- **S-IoU 只是可比性數字** —— 身為閾值正負號,它會像 near-sign 一樣飽和;**別**拿它當閘門。
- **Chamfer / NC / F-score** 是*連續*的細表面判準(無 sign floor)—— 它們承載「表面到底有沒有學到」這個問題。
- **τ 必須超過 MC cell 大小** `2/res`(res 128 時 ≈0.0156),否則 F-score 會評比比一個 vertex 能擺放的精度還細的一致性。
- **μ-path vs sample-avg。** decode(μ) 不被 ELBO 保證;在高維 latent 中樣本活在半徑 σ√d 的殼上,而 μ 可能是 decoder 從不訓練到的近零質量中心。後驗 μ-broken 時 μ-path 會嚴重低估甚至崩壞 → **一律以 sample-avg(K=32)為準**,μ-path 只當診斷。
- **退役指標:** **near-sign-acc** = near-band 點裡預測正負號正確的比例(在可解析精度之下飽和 → 作為目標被丟棄);**RMS** = √(mean SDF-error²),與 `gt` 同單位(當訓練健康度讀數用;PASS 門檻 `RMS<0.02` = near-band 寬度)。

## 論文

### 3D Shape VAE
| 論文 | 使用的貢獻 |
|---|---|
| 3DShape2VecSet(SIGGRAPH 23) | VecSet 表徵;occupancy+BCE 監督;KL=1e-3,明說是為生成階段 |
| Hunyuan3D-2 / 2.1 | VAE-DiT 結構;ShapeVAE latent×64、enc8/dec16、width 1024;`encode` 預設 `sample_posterior=True`(解碼樣本,不是 μ)。**Token 數未解:日誌記 4096,但論文的重建實驗用 1024 —— 確認哪個是 recon VAE config。** |
| Dora | 銳邊採樣 |
| DeepSDF(CVPR 19) | Clamped-L1 SDF loss δ=0.1;**auto-decoder**(per-shape latent,無 encoder/KL → 無塌陷 —— 為何 clamp 在那裡安全但在我們的 encoder+KL VAE 致命) |

### DiT / Diffusion
說明: diffusion 模型在前期雕形狀後期雕細節

|論文|	arXiv	|貢獻|
|---|---|---|
|DP-DMD (Diversity-Preserved DMD)	|2602.03139	|在 SD3.5-M(rectified flow)上直接觀察去噪軌跡,附錄 B + Figure A|
|TIDE	|2503.07050|	用 sparse autoencoder 分析 DiT,找出輪廓形成的轉折timestep|
|SuperEdit	|2505.02370	|四階段分解:早期全域佈局 / 中期局部物件屬性 / 後期細節 / style 橫跨全程|
|UniTransfer|	2509.21086	|Chain-of-Prompt,按 coarse/medium/fine 三級把 prompt 分階段注入|
|SCoPE (Progressive Prompt Detailing)|	CVPR-25 workshop	|漸進式 prompt 細化,依 coarse-to-fine 假設設計|
|SPARE	|2602.07058|	Timestep Targeting 一節引用此性質|
|DiT-BlockSkip	|2603.20755	|依 timestep 調整 patch size,基於高 timestep 學全域、低 timestep 學細節|
|Mixture-of-Diffusers	|OpenReview lcmd2Qdrsv	|時間序列領域,early/late-stage diffusers 分工|

*表格中多數是引用此性質來設計方法,而非專門驗證它的論文。真正做系統性測量的是 TIDE 和 DP-DMD 附錄。
全部是影像/影片,沒有 3D。

## Backlog / 開放問題

**重跑清單(一律用論文官方 in-distribution split;舊切分數字已移除,不可信)。**
- ✅ `vae_baseline_1024_l8_16`(plain,20 ep)—— 已訓練 + 部分 eval(見 Stage-2);待補完整 sample-avg(256 SDF + 16 mesh)。
- ✅ `anchor_baseline_1024_l8_16`(anchor,20 ep)—— 已訓練 + SDF eval(見上);待補 mesh。
- ✅ `anchor_converge_1024_l8_16`(kl1e-3 訓到收斂,lr÷10 沉定)—— 已 eval(見 Converge run:volume-IoU 88.8%、F-score 0.979);`max_epochs=-1` 可隨時 resume 續訓。
- ⏳ 待重跑的消融:capacity 對照(`l8_22`,純 depth)、detach(切斷 anchor→encoder 梯度)、KL sweep、arch-only(anchor 架構但 weight=0)。

**開放問題。**
- 確認論文的 1024 token 對應哪一個 recon VAE config。
- baseline / anchor 一律以 **sample-avg** 評估(μ-path 對 μ-broken 後驗不可信);補齊對等的 shape 數與 mesh 指標,才能正式比較。
- KL 旋鈕:free-bits vs warmup vs 固定 `kl_weight` 對 μ-可用性與 recon 銳利度的影響。
- Augmentation(rotation/jitter;SDF 剛體等變)對小資料泛化是否有幫助?
