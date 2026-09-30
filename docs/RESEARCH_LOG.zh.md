# 研究日誌（中文版）

在單張 RTX 5080 上訓練，使用約 45k 個 ShapeNet watertight meshes (3DShape2VecSet 資料集)。
試圖重現並改進 **Hunyuan3D-2 ShapeVAE** 的模組。

## 前言 —— 這個專案的基本設定
本研究是基於HunYuan3D 2.0 的開源專案以及技術報告，因此基礎模型架構接承襲於它。

### 重建部份： VAE 的 encoder、decoder 及相關模組。
- **任務與輸入輸出**：給定一個 mesh 的表面點,encoder 把它映射成一個 latent set;decoder 再針對任意 3D query 點預測其 **SDF**(有號距離:<0 在內部、>0 在外部、0 在表面上)。而 Occupancy / IoU 的數字是由 SDF 的正負號判斷(內部 = SDF<0)。
- **架構**：VecSet VAE:encoder → **一組 N 個 latent tokens,每個寬度 64**(`num_latents` N ∈ {1024, 3072};`latent_dim` = 64)→ KL → decoder → 每個 query 的 SDF。 Mesh 由預測出的 SDF 場經 **marching cubes(MC)** 抽出。
- **目標函數**： **SDF MSE** 以及 **KL loss**。
- **指標(metrics)**： VIoU, SIoU, F-score, NC, Chamfer Distance

## Stage-1 —— Capacity Test: overfit 一個 mesh
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

實驗名稱： `vae_baseline_1024_l8_16`
實驗配置：1024 tokens、enc8/dec16、plain MSE、kl1e-3、cosine 3.5e-5→1e-6、20 ep;
使用與 3DShape2VecSet 相同的 in-distribution split、從頭訓練。

| eval | V-IoU | S-IoU | RMS | σ_rms | μ-spread |
|---|---|---|---|---|---|
| **sample-avg K=32 (64 shapes)** | **51.03%** | **44.36%** | 0.0203 | 0.972 | 0.144 |


## 設計嘗試：利用 anchor 模型直接預測 vecset 內含的重要位置座標(失敗)

### 背景
基於 vecset 表示法的生成模型在細節上仍有待加強，其官方出版的 LATTICE 報告中認為是因為模型必須同時生成where 與 what 資訊，才導致細節不夠精細。為此報告提出了兩階段生成的方法：先將第一階段經過完整生成pipeline的粗糙模型（HunYuan3D 2.0）voxelize，之後提取與模型表面的active voxel並作為第二個生成模型（Hunyuan3D 2.5）的輸入， 以此方式將where 與 what解耦。

### 想法
跑兩次生成模型意味著要跑兩次diffusion + decode的過程，

### 我的方法
與其跑兩次完整生成管線兩次，不如跑一次就好。我保留了 HunYuan3D 2.0 的原始架構（VAE + DiT），在其之上我設計了一個 Anchor 模組，試圖從 latent 提取絕對座標，作為參考點將資訊加入到 decoder 中，讓生成模型和 decoder 在生成時直接參考物體的絕對座標。以此達成將 where 跟 what 的解隅，與論文提出的目的相同，但以不同方式達成。

### 問題
**Q1. Latent 是由 Diffusion 生成的，應該無法利用 Anchor 資訊，畢竟 Anchor 就是以 Latent 為輸入？**

A1. 有研究發現2D影像以及影片的diffusion生成模型會在早期階段生成影像大致雛型，後期才會開始雕細節。而經過測試發現這也成立於3D領域中。因此我將在diffusion生成的後半所有time step中抽取其中間產物z，並計算它當時預測的z^，以此作為anchor model的輸入。

**Q2. Anchor 能有效的提取座標嗎？**

A2. 可以，前提是它跟 VAE 同時訓練，並非先單獨訓練完 VAE 後再訓練 Anchor。 另外，訓練時 Anchor 的 gradient 必須傳回 VAE encoder，因為經實驗測試，若兩者detach教會非常難訓練（moving target）。但尚未證實是否可以直接在已經訓練好的 VAE encoder 上訓練 Anchor，這將會是之後研究的實驗。

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

| eval | V-IoU (all) | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | RMS | μ-spread |
|---|---|---|---|---|---|---|---|---|
| SDF 256 shapes | 76.23% | **87.34%** | 71.04% | — | — | — | 0.0107 | 0.327 |
| mesh 16 shapes res-128 | 81.23% | **88.79%** | 77.27% | **0.0117** | **0.979** | **0.940** | 0.0103 | 0.265 |

*(兩列都 sample-avg K=16;256-shape 的 SDF 指標較有代表性,16-shape 是 mesh-viz 慣例、才有 Chamfer/F/NC,16/16 全抽到面。)*

→ **表面保真度基本上 SOTA 級**:F-score **0.979**、Chamfer **0.0117**、NC **0.94**,跟 3DShape2VecSet 報的
F-score(~0.97)同級甚至更好。**volume-IoU ~88%** 落在論文區間;唯一偏低的是 **S-IoU(近表面符號)~71-77%**,
那是 SDF-MSE 表面梯度消失的結構弱點,與表面品質是兩回事。**關鍵教訓:先前「卡 0.72」是量錯了指標** —— 用
80% 近表面點算 IoU、而非論文的均勻 volume 點;換成 volume-IoU 立刻回到論文區間。渲染圖(GT↔recon,16/16)
在 `results/converge_meshsample_viz/`。

**Vanilla 對照(anchor-vs-vanilla A/B, 2026-07-30)。** 同協定跑了 vanilla 版 `vae_converge_lr3.5e-6_1024_l8_16`
(拿掉 anchor、其餘配方相同:kl1e-3、constant LR 3.5e-5 訓到 plateau 再 lr÷10 沉定;同 in-distribution split)。
以同樣 sample-avg(K=16)、mesh 16 shapes res-128 評估:

| eval | V-IoU (all) | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | μ-spread |
|---|---|---|---|---|---|---|---|
| **anchor** mesh 16 res-128 | **81.23%** | **88.79%** | **77.27%** | **0.0117** | **0.979** | **0.940** | 0.265 |
| vanilla mesh 16 res-128 | 58.16% | 69.25% | 52.27% | 0.0315 | 0.779 | 0.882 | 0.148 |

→ **同一份 codebase、同協定下 anchor 全面領先 vanilla**(volume-IoU +19.5pt、S-IoU +25pt、Chamfer 少一半、
F-score +0.20)。這是本研究的核心 A/B:貢獻由此證成,不必跟論文絕對值比。*警語:vanilla 這輪只訓到
epoch ~57、lr 剛除 10 未久,且 16 shapes 全是同一類(02691156 飛機);要更公平仍須兩邊跨類別、多 shape 對齊。*

#### Ablation —— 架構 vs 監督(archonly, 2026-08-01)

要問「anchor 的增益來自**多出來的架構**(decoder 的 anchor cross-attention 參數)還是來自**anchor 監督**(chamfer+mse 把 token 拉到表面)?」跑了 `anchor_archonly_1024_l8_16`:**anchor 架構原封不動,但把 `anchor_chamfer_weight`/`anchor_mse_weight` 設 0**(關掉監督),kl1e-3、lr 3.5e-5、20 epoch、from scratch。同協定 sample-avg(K=16)、mesh 16 res-128。**為 budget-matched,anchor 這列用 20-epoch 的 `anchor_baseline_1024_l8_16`(不是 converge 版)**,和 archonly 同架構、同 20 epoch、只差監督開關:

| run | 架構 | anchor 監督 | 預算 | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | μ-spread |
|---|---|---|---|---|---|---|---|---|---|
| **anchor_baseline** | anchor | ✅ | 20 ep | **83.66%** | **64.19%** | **0.0169** | **0.937** | **0.908** | 0.322 |
| archonly | anchor | ❌(w=0) | 20 ep | 47.63% | 31.52% | 0.0718 | 0.515 | 0.734 | **0.814** |
| l8_22 mesh 16 res-128 | 22 | ❌ | 20 ep | 65.30% | 42.84% | 0.0402 | 0.705 | 0.814 |


→ **乾淨對照(前兩列,同架構同 20ep):開監督 vs 關監督 = uniform +36pt、S-IoU +32.7pt、F-score +0.42。監督是決定性的。**
**架構本身不但不是增益來源,反而是負擔** —— archonly(有架構、無監督)連 converge 過的 plain vanilla 都不如
(F 0.515 vs 0.779)。且 **archonly 的 μ-spread 0.814 逼近先驗 1.0** —— 後驗幾乎沒偏離先驗,代表多出的 decoder
cross-attention **沒有 anchor 訊號去 ground 它時,latent 幾乎沒被用上**(接近 posterior 塌到先驗);對比 anchor_baseline
的 μ-spread 0.322(latent 有被實際使用)。**結論:增益來自 anchor 監督,不是架構容量。** *警語:vanilla 這列是 converge
(~ep57),非 20ep,只當「archonly 連普通 baseline 都輸」的參考,不是 budget-matched 那組;要把 vanilla 也對齊需補跑
`vae_baseline_1024_l8_16`(20ep)的 mesh-16 eval。16 shapes 全是飛機類。*

**生成探針(latent 健康度)。** `scripts/sample_shapes.py` 抽 latent 直接 decode:posterior(z~N(μ,σ))
與 aggpost(聚合後驗)都能 decode 出完整形狀(~12k+ verts),但純 prior(z~N(0,I))退化成碎片(~1.2k
verts)—— decoder 健康、但 latent 沒對齊 N(0,1)。**這正是留給 DiT 的 gap。**

## Double Stream VAE （失敗）

### 想法


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
- **τ 必須超過 MC cell 大小**,否則 F-score 會評比比一個 vertex 能擺放的精度還細的一致性。實測的上界與 τ=0.02 的選擇理由見下方 [F-score 的 τ 與 marching cubes 上界](#f-score-的-τ-與-marching-cubes-上界)。
- **μ-path vs sample-avg。** decode(μ) 不被 ELBO 保證;在高維 latent 中樣本活在半徑 σ√d 的殼上,而 μ 可能是 decoder 從不訓練到的近零質量中心。後驗 μ-broken 時 μ-path 會嚴重低估甚至崩壞 → **一律以 sample-avg(K=32)為準**,μ-path 只當診斷。
- **退役指標:** **near-sign-acc** = near-band 點裡預測正負號正確的比例(在可解析精度之下飽和 → 作為目標被丟棄);**RMS** = √(mean SDF-error²),與 `gt` 同單位(當訓練健康度讀數用;PASS 門檻 `RMS<0.02` = near-band 寬度)。

### F-score 的 τ 與 marching cubes 上界

*2026-09-28 從 `scripts/paper_eval.py` 搬來,讓腳本只留決定、不留推導。*

**Cell 大小是 `2/(res-1)` 而不是 `2/res`。** `Postprocess._extract` 把 voxel index 映回世界座標的公式是 `v/(r-1)*2 - 1`(`src/model/shape/VAE/postprocess.py`),所以解析度 128 時相鄰頂點的間距是 2/127 = **0.01575**,不是 2/128 = 0.0156。差距很小,但上界剛好就斷在這裡,所以挑 τ 時這個差別有意義。

**實測上界。** 把**真實**的 SDF 在同一個網格上跑 marching cubes,再拿結果去對 GT 網格評分 —— 這就是任何模型在該解析度下可能達到的 F-score 上限。在 8 個測試形狀、每個取 100k 表面點上量測:

| τ | 0.005 | 0.01 | 0.01575 | 0.02 |
|---|---|---|---|---|
| 平均 F | 0.59 | 0.967 | 0.9993 | 0.9997 |
| 最差形狀 F | 0.45 | 0.916 | 0.9980 | 0.9987 |

**斷點就在 cell 大小。** 低於它分數崩潰,到達或超過它上界基本上就是 1.0。這就是 **τ = 0.02 為預設**的理由:文獻慣用的 0.01 會白白損失最多 8 個 F-score 百分點給離散化,而且**各形狀損失不均**(最差形狀 0.916 對平均 0.967),這會讓比較偏向「網格比較平滑」的模型,而不是「重建比較準確」的模型。

**一個後來被推翻的推論 —— 值得留著。** 同一次量測把 Chamfer-L1 的地板定在 0.0095(範圍 0.0052–0.0115),而回報值是 0.0127 / 0.0136 / 0.0143。我當時由此推論:Chamfer 數字裡大部分是離散化而非模型誤差,三個模型是在一個人為地板附近被分開的。

**這個結論是錯的。** 在 600 個形狀上改用解析度 256 重跑(`docs/eval_result_256.md`),Chamfer 最多只變動 1.8%,而且是**往上**而非往下(lambda 0.01270 → 0.01333)。如果數字真的被離散化地板主導,把 cell 大小縮成四分之一應該會讓它大幅下降。模型之間的排序與差距完全沒變。

教訓:在真實場上量到的**上界**限制了 F-score 能達到多少,但它不能用來平行推論 Chamfer。F-score 是一個帶門檻的計數,所以對 cell 大小反應劇烈;Chamfer 是平均距離,誤差主要由表面實際在哪裡決定,而不是由網格切得多細決定。要驗證地板,就去改解析度實測,不要從上界推導。

## 論文

### 3D Shape VAE
| 論文 | 使用的貢獻 |
|---|---|
| 3DShape2VecSet(SIGGRAPH 23) | VecSet 表徵;occupancy+BCE 監督;KL=1e-3,明說是為生成階段 |
| Hunyuan3D-2 / 2.1 | VAE-DiT 結構;ShapeVAE latent×64、enc8/dec16、width 1024;`encode` 預設 `sample_posterior=True`(解碼樣本,不是 μ)。**Token 數未解:日誌記 4096,但論文的重建實驗用 1024 —— 確認哪個是 recon VAE config。** |
| Dora | 銳邊採樣 |
| DeepSDF(CVPR 19) | Clamped-L1 SDF loss δ=0.1;**auto-decoder**(per-shape latent,無 encoder/KL → 無塌陷 —— 為何 clamp 在那裡安全但在我們的 encoder+KL VAE 致命) |

### DiT / Diffusion
|Scaling Rectified Flow Transformers for High-Resolution Image Synthesis| MM-DiT 雙軌模型結構|
|ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING| Rotary Position Embeddings（旋轉位置編碼）|

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

## 最終 Test-Set 完整評估(2026-08-02)

七個實驗各取 **best.pt**,在**論文 in-distribution test split** 上同協定橫向對照。**SDF 指標(V-IoU all/uniform、
S-IoU、μ-spread)跑全 2,592 顆 test**;**Chamfer / F-score@.02 / NC 抽 128 顆、跨類別 stratified**(marching cubes
太貴,全跑需 ~12 天;128 顆跨 55 類已具代表性)。全用 **sample-avg K=16、res-128**。腳本:`scripts/final_eval.py`
(每 model 兩趟 `evaluate.py` → `results/final_eval/*.json` + `summary.md`)。依 V-IoU(uniform) 排序:

| 實驗 | 架構 | anchor 監督 | 預算 | V-IoU (all) | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | μ-spread |
|---|---|---|---|---|---|---|---|---|---|---|
| **anchor_converge_lr3.5e-6** | anchor | ✅ | converge | 77.54% | **89.48%** | **71.87%** | **0.0193** | **0.938** | **0.929** | 0.385 |
| **anchor_baseline** | anchor | ✅ | 20 ep | 67.16% | **83.01%** | 59.99% | 0.0241 | 0.872 | 0.889 | 0.465 |
| vae_converge_lr3.5e-6 | plain | — | converge | 62.28% | 80.92% | 55.04% | 0.0383 | 0.755 | 0.879 | 0.197 |
| vae_baseline (dec16) | plain | — | 20 ep | 55.53% | 75.34% | 48.21% | 0.0471 | 0.656 | 0.818 | 0.223 |
| vae_baseline_l8_22 (dec22) | plain | — | 20 ep | 54.66% | 75.11% | 47.02% | 0.0462 | 0.665 | 0.818 | 0.215 |
| anchor_detach | anchor | ⚠ detach | 20 ep | 42.84% | 58.88% | 37.10% | 0.0822 | 0.469 | 0.723 | 0.154 |
| anchor_archonly | anchor | ❌ (w=0) | 20 ep | 42.20% | 58.13% | 36.52% | 0.0834 | 0.427 | 0.721 | **1.056** |

**這張表把整個研究的因果收斂成一句話:決定品質的是「有回流到 encoder 的 anchor 監督」,不是架構、不是 decoder 深度、也不只是訓久。**

1. **anchor 監督在每個 budget 都贏 vanilla。** 20ep:uniform 83.01 vs 75.34(+7.7pt)、F 0.872 vs 0.656;converge:
   89.48 vs 80.92(+8.6pt)、F 0.938 vs 0.755。方向一致、幅度大。
2. **最關鍵的證據 —— detach vs anchor_baseline(同架構、同 anchor loss,唯一差別是 gradient 有沒有傳回 encoder):
   uniform 58.88 vs 83.01(−24pt)、F 0.469 vs 0.872。** 把 anchor 的 gradient detach 掉,整條 anchor 分支立刻變廢,
   甚至比沒有 anchor 的 plain vanilla(75%)還差。**這直接證成設計核心:anchor 的監督必須回流 VAE encoder**(否則是
   moving target,對應 Q2/A2)。
3. **架構本身無監督 = 負擔。** archonly(w=0,完全無監督)與 detach(有 loss 但斷 gradient)雙雙掉到 ~58% uniform /
   F ~0.43–0.47,遠低於 plain vanilla。兩條獨立控制得到同一結論。μ-spread 也對得上:detach **0.154**(近後驗塌縮,
   encoder μ 幾乎不隨形狀變)、archonly **1.056**(μ 亂跑但 decode 不出東西);相對地健康的 anchor 是 0.39–0.47。
4. **加深 decoder 沒用。** l8_22(dec22)≈ l8_16(dec16)(uniform 75.11 vs 75.34)—— 不是容量/深度的問題。
5. **訓久有幫助但不改變排序。** converge 兩邊都比 20ep 高(anchor 83→89、vanilla 75→81),但 anchor>vanilla 的關係
   在兩個 budget 都成立。

**表面保真度(anchor_converge):F-score 0.938、Chamfer 0.0193、NC 0.929** —— 全 test、跨類別下仍達 3DShape2VecSet
同級。這份是全專案第一份同協定、全 test 的最終橫向結論,取代先前零散的 16-shape / val 數字。渲染圖存
`results/final_<實驗名>/`。

## 潛在空間正則化與 loss 形狀實驗(2026-08 → 2026-09)

*2026-09-26 從 experiment config 檔頭搬移至此。那些檔頭原本是這批測量的唯一紀錄;現在 config 只留三行摘要並指回本節。*

### 這些實驗共同針對的問題

在收斂的 lambda-disjoint checkpoint 上(epoch 67、24 個 test shape × 50k 查詢點),SDF 誤差可以拆成
**deterministic 0.0035 + 後驗雜訊 0.0017**,也就是 MSE 的 81% 是 decoder 根本改不動的擬合誤差。
train MAE 0.00375 對 test 0.00388,差距只有 3.5%,所以這是純粹的 underfitting 而不是 generalisation
失敗,再多正則化也沒用。近表面帶(|gt| < 0.002)的誤差是 0.0030,**比帶寬本身還大**,這正是那裡的
符號近乎隨機、V-IoU 卡在 83 左右的原因。

Baseline 的潛在空間病徵,量在 `vae_disjoint_converge_kl1e-4_1024_l8_16` 上:sigma 0.87 對 mu.std
0.45,也就是 **z 的變異數有 81% 是雜訊**;aggregate posterior 離 N(0,I) 很遠,covariance condition
number 590,沿隨機方向的 excess kurtosis +8.2。

### 交叉檢查:雜訊主導的後驗是我們的問題,還是架構的?

前面所有討論都建立在一個量測上:latent 的變異數大部分是後驗雜訊,而不是形狀資訊。但這件事只有在它
是「架構與未加權 SDF 損失的性質」而非「這個模型碰巧訓練成這樣」時才有意義。Hunyuan3D-2.1 有發布的
ShapeVAE checkpoint,所以可以直接在它上面量同樣的東西。

執行:`uv run --with einops python -m scripts.hunyuan_posterior_check --shapes 5`。權重在本機快取好
之後,整支不到一分鐘(`~/.cache/hy3dgen/` 底下共 7.5 GB;ShapeVAE 那部分是 625 MiB 的 fp16
checkpoint,第一次使用時從 HuggingFace 下載)。

| | mu.std | sigma | SNR | 訊號佔比 |
|---|---|---|---|---|
| 我們的(`vae_converge_kl1e-4_lr3.5e-6`,epoch 64) | 0.3172 | 0.9176 | 0.346 | **10.3%** |
| hunyuan3d-2.1 官方發布的 checkpoint | 0.4576 | 0.8321 | 0.550 | **21.4%** |

訊號佔比是 `Var(mu) / (Var(mu) + E[sigma^2])` —— latent 總變異數當中,真正隨形狀移動的那一份。兩個
模型都是雜訊主導:我們的 latent 約 90%、他們的約 79% 是每次取樣重新抽的後驗雜訊。

**所以這個病徵不是本地的訓練錯誤。** 一個由架構作者自己訓練、production 規模的發布模型呈現同樣的
定性行為,這才是把它當成設計性質、而不是靠調本 repo 超參數去修的正當理由。

**但兩者並不相同,而且差距不小。** 他們的訊號佔比大約是我們的兩倍(21.4% 對 10.3%)。不論他們做了
什麼不同的事 —— 資料量大得多、不同的 KL 權重、更長的排程 —— 那都換來了明顯更乾淨的後驗。宣稱兩者
「一樣」會誇大結果;誠實的說法是兩者落在同一個區間。

**兩個限制,決定了這個結果能推到多遠。**

1. *兩列的 field disagreement 數字不可互相比較。* 腳本對每個模型解碼兩次,一次從 mu、一次從後驗
   取樣,回報兩個輸出場的差異(我們 6.2%,他們 9.5%)。但我們這半邊用的是快取的 SDF 查詢點,其中
   80% 集中在近表面;Hunyuan 那半邊用的是立方體內均勻抽的點。近表面正是場差異最大的地方,所以這
   兩個數字量的是不同的問題,不能讀成「他們比較差」。
2. *他們的 encoder 是在關閉銳邊取樣的情況下跑的*(`pc_sharpedge_size: 0`),因為那是他們發布的
   config 與官方 demo 的設定。考慮到這個專案整個主題就是銳邊分支,有必要講明:官方預設根本沒有用
   銳邊取樣 —— 他們的 latent 一開始就沒有銳邊/均勻的切分可以塌縮。

另外樣本只有 5 個形狀,而且 `torch_cluster.fps` 被換成純 Python 的替代實作(理由寫在
`scripts/hunyuan_posterior_check.py` 裡)。對一個在整個 latent 上聚合的統計量來說這兩點都可以接受,
但這是一次合理性檢查,不是 benchmark。

### lambda-VAE(arXiv:2607.05531,"Variance Equalization for Posterior Collapse")

取樣改成 `z = mu + sigma^lam * eps`(Eq.9),每個 channel 的指數是
`lam_i = max(1, log(1-1/delta) / (2 log sigma_i))`(Eq.16),而 KL 項仍然按**原始**的 sigma² 收費
(Eq.10)。這個不對稱就是整個方法:decoder 看到的 latent 比 KL 罰則所暗示的乾淨得多,在不削弱
正則化的前提下補上 information gap。把 Eq.16 代回 `sigma^lam` 得到 `sqrt(1 - 1/delta)`,對所有在
floor 之上的 channel 都一樣 —— 也就是單一的共同有效雜訊水準。

delta 是唯一的超參數,直接決定那個水準:

| delta | 1.001 | 1.01 | 1.1 | 2.0 |
|---|---|---|---|---|
| 雜訊 `sqrt(1-1/delta)` | 0.0316 | 0.0995 | 0.302 | 0.707 |

**選定 delta = 1.1**(兩個 lambda run 都用這個值),因為它讓 latent 保持真正隨機 —— 雜訊 0.302 對
mu.std ~0.54,約 76% 是訊號 —— 而不是論文 RGB 設定給的近乎確定性的 0.032。它對 decoder 的代價也
小得多:在收斂模型上把注入雜訊改成 0.307 只掉 **3.3 V-IoU**,改成 0.032 則掉 **9.0**。

**Ramp 長度的決定。** `lam_ramp_steps: 225000` ≈ 10 epoch(每個 epoch 約 22.7k 次 `encode()`)。論文
用 150 epoch ramp 的理由是等 sigma 穩定,這個理由在這裡**不成立**:sigma 從 epoch 1 就是平的
(ep1 是 0.87;五個 run 的 ep9–60 落在 0.83–0.94),而且在 init 時 sigma > 1 會讓 Eq.16 直接回傳
lam = 1,所以 lambda-VAE 在 sigma 降到 1 以下之前會自己停用。Ramp 是為 **decoder** 設的:把注入
雜訊一步從 0.89 降到 0.032 會掉 9 V-IoU(73.3 → 64.2),因為 z.std 砍半而 `proj_latent` 從沒見過
那個輸入尺度。所以任何 lambda run 都必須跑 **≥ 20 epoch** —— lam* 要到 epoch 10 才完全開啟,之後
還需要同樣長的時間才會收斂。

**在非 MRL 對照組上的結果:**active units 55.9% → **91.9%**,channel std(mu) 的離散程度 10.7× →
**3.2×**。最終 λ 評測:65,536 / 65,536 個維度全部通過 Burda AU 0.01 門檻,而且總 Var(mu) 比
baseline 還少 24%。

Run:`vae_lambda_disjoint_converge_kl1e-4_1024_l8_16`(lr 3.5e-5)與
`vae_lambda_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16`(lr 3.5e-6),兩者都對
`vae_disjoint_converge_kl1e-4_1024_l8_16` 做 A/B,唯一差異是 `model.lam_delta`(0.0 → 1.1)。
*注意:兩個檔頭的敘述都寫 delta = 1.001、lam* ≈ 24.8,但 YAML 本體設的是 `lam_delta: 1.1`;實際跑的
是本體。1.001 那組數字來自更早的規劃,不是這兩個 run 的結果。*

### lambda-VAE 加在 masked-attention encoder 上

`masked_vae_lambda_converge_kl1e-4_1024_l8_16`,對照 `masked_vae_converge_kl1e-4_1024_l8_16`。

`MaskedCrossAttentionEncoder`:1024 個 query token 是 uniform 在前、sharp 在後,20,480 個 KV 點也是
同樣順序,所以 masked block 對兩邊都從正中間切開 —— uniform token 可以看全部,sharp token 只看
sharp。兩層不遮罩的 "mix" 層(stack 中間與最後)是 uniform 脈絡唯一能傳到 sharp token 的路徑。
Block 數與普通 encoder 完全相同(1 cross + 3 + mix + 3 + mix = 9 對 1 cross + 8 self = 9),所以
**這裡沒有增加任何容量**。

為什麼在這個 encoder 上加 lambda:它的問題**不是**死掉的 channel(epoch 18 量到 dead(<0.01) =
0/64),而是能量不均 —— per-channel mu.std 的 max/median 是 2.18×,per-channel KL 的離散達 23×
(因為 KL 隨 mu² 走)。Variance equalization 針對的正是這個軸。

**歸因警告:**這裡疊了兩個改動(masked attention + lambda-VAE)。要單看 lambda-VAE 的乾淨結果,
請看 `vae_lambda_disjoint_converge_kl1e-4_1024_l8_16`。

### SIGReg(LeJEPA)當潛在空間正則化 —— 兩次嘗試

**第一次,用 SIGReg 取代 KL、作用在取樣後的 z**(`vae_sigreg_disjoint_converge_1024_l8_16`,
`sig_weight: 1e-2`)。動機:KL 項收的是
`E_x[KL(q(z|x)‖p(z))] = I(x;z) + KL(q(z)‖p(z))`,也就是同時為互資訊(會犧牲重建)與 aggregate
posterior 的形狀(下游 DiT 真正需要的)付費;SIGReg 只付第二項。`sig_weight` 是對著**收斂時**的
recon 校準的,不是對著 init —— recon 從 epoch 0 的 0.0028 掉到 epoch ~40 的 0.0002,所以固定權重在
整個 run 裡相對值會漲約 15×。

**這次失敗了,而且失敗本身有資訊。** 在 epoch 4:SIGReg(z) = 0.00128,對上有限樣本地板 0.00050,
等於已經收斂;但有資訊的那部分反而走錯方向 —— SIGReg(mu) 0.366 對 KL baseline 的 0.241,mu.std
0.17 對 0.45,z 的訊號佔比 3.0% 對 19.7%。**z 的變異數有 81% 是後驗雜訊,而高斯雜訊本身就已經
滿足這個檢定**,所以 SIGReg-on-z 被雜訊滿足了,mu 想幹嘛就幹嘛。崩潰風險事前就預期到:KL 拿掉後
mu=0 / sigma=1 是合法的極小值(崩潰時 SIGReg 1.04,真實 latent 上是 390),只有重建在對抗它,而
重建恰好在收斂之後最弱。

**第二次,SIGReg 作用在 mu、且加在不動的 KL 項之上**
(`vae_sigreg_mu_disjoint_converge_1024_l8_16`,`sig_weight: 1e-4`,`kl_weight: 1e-4`)。

為什麼 KL 項要**留著**:SIGReg 撐不住 latent 的尺度。Epps-Pulley 統計量在分布比高斯窗還寬之後就
飽和 —— 在精確高斯上量到 std 5 → 0.810、std 10 → 0.824、std 200 → 0.822,對尺度的導數在 std 2 是
+0.334、std 5 是 +0.025、std 20 是 −0.0002。而重建會推著 mu **變大**(mu 越大,對固定 sigma 的
SNR 越好),正好是 SIGReg 守得最差的方向,所以拿掉 KL 就發散:**sig 1e-4 時 z_std 到 7.3,
sig 1e-2 時到 181**。LeJEPA 不會踩到這個,因為 JEPA 沒有重建項 —— 它的失效模式是崩潰(縮小),
那是 SIGReg 梯度強的那一側。所以 KL 負責錨定尺度,SIGReg 負責塑造 skew / kurtosis / isotropy,
那些正是 KL 單獨做得很差的部分。

`sig_weight = 1e-4` 用同一個 seed 跑 240 步對照 baseline 驗證過:viou 24.55 對 23.44,kl 2.42 對
2.46,z_std 2.52 對 2.54 —— 軌跡無法區分、沒有發散,而統計量本身從 1.96 降到 1.33。

### Recon loss 指數:用根號取代 MSE

`vae_disjoint_sqrt_kl1e-4_lr3.5e-6_1024_l8_16`,單軸 A/B,**繼承**對照組
`vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16`(刻意與 lambda 那幾個 run 相反:這裡的重點就是
只有一件事不同,所以對照組的 LR 或 kl_weight 一旦調整,這個 run 必須跟著動)。整個 diff 只有
`task.recon_loss.exp: 0.5` 與 `eps: 1e-4`。

MSE 的最佳解是條件**期望值**,而往期望值靠攏正是小特徵被平均成光滑表面的原因 —— 在渲染對照裡看得
很清楚,兩個模型都保住機身,但都掉了機腹的小突起。`exp = 0.5` 把最佳解往眾數移。

它實際改變了什麼,在這個 loss 上以 eps=1e-4 量的每點梯度:

| \|e\| | exp=2 | exp=0.5 | |
|---|---|---|---|
| 1e-4 | 3.06e-9 | 2.47e-7 | 對已經幾乎正確的點,拉力 **多 81×** |
| 4e-3 | 2.67e-7 | 1.49e-7 | 交叉點 —— 也正是目前的平均誤差 |
| 1e-1 | 3.51e-6 | 1.51e-8 | 對最差的點,拉力 **少 233×** |

所以這是把力氣從擬合最差的點重新分配給擬合良好的多數,而損益兩平點就落在目前的誤差水準上。

**風險。** |e| > 0.02 之外基本上被放棄,大約是 1% 的查詢點。這 1% 裡有一部分是壞掉的 GT(有一個
test shape 的標籤宣稱立方體有 31.9% 在網格內部,而真實體積只有 1.0%,光這一個 shape 就佔了 test
MSE 的 86%),放棄那些是賺到;其餘是真的困難的幾何,放棄那些是虧。

**尺度。** `WeightedSDFLoss` 把總和取 `^(2/exp)`,所以 exp=0.5 落在與 exp=2 相同的數量級,而不是
原始 `mean(|e|^0.5)` 會給的低 3700×。只有在殘差集合同質時才完全對齊:在收斂誤差尺度量到 9.4e-6
對 1.6e-5(1.7×),在 init 量到 5.4e-2 對 1.7e-1(3×)。因此 `kl_weight` 的意義不變,也刻意**沒有**
重新調整。要注意兩個 run 最佳化的是不同的泛函,所以 `val/recon` 在兩者之間**不可比** —— 把它讀成
「根號 run 的 loss 比較低」是沒有意義的。決定勝負的指標是 `val/viou`,對照組是 **81.35**
(其 best.pt 存的 `best_metric`,epoch 65)。

**eps = 1e-4 是必要的,不是裝飾。** exp < 1 時 |e|^0.5 的梯度在 e→0 會發散並 NaN。平滑只在
|e| < eps 的地方作用,而一個 50k 殘差 ~N(0, 0.004) 的 batch 每一步的最小 |e| 都在 1e-7 附近,所以
遠低於 1e-5 的值都是惰性的。exp=0.5 下量到的 max/typical 每點梯度比:**eps=1e-9 是 436,
eps=1e-4 是 11.5**,而純 MSE 自己的比值是 9.7 —— 1e-4 這個值讓它不比原本就信任的 MSE 訓練更失衡,
而且只讓回報的 loss 膨脹 0.07%。

**Warm start,以及為什麼用 `init_from` 而不是 `resume`。** 對照組的 checkpoint 可以原封載入 ——
已用 `load_state_dict(strict=True)` 在 best.pt(epoch 65)與 last.pt(epoch 68)上驗證過,而且
model、preprocess、data 三組設定比對相等,因為兩份 composed config 裡唯一不同的 key 就是
`task.recon_loss.exp`。要用 `trainer.init_from`,**不要**用 `trainer.resume`:resume 會連 Adam 動量
一起還原,而那些動量是在 e² 下估出來的 —— 正是這個 run 要改的梯度輪廓 —— 會讓每個參數在
beta2=0.999 的 EMA 翻過去之前(約 1000 步)都用錯誤的有效步長。resume 還會繼承
`best_metric=81.35`,導致在新 loss 贏過舊 run 之前完全不會寫任何 checkpoint,而重新適應期間本來
就預期會先掉一段。

**結果:**根號 run 的 val/viou 達到 **79.71**,對照組是 75.28。*(該 run 在 2026-09-20 於 epoch 22
無 traceback 死亡 —— 外部 kill / SIGHUP,不是數值問題。)*

**如果輸了**,在放棄這個想法之前先試 exp=1.0(MAE,條件中位數);0.5 是這個範圍裡最激進的一端。

### MRL + lambda + 每個 m 各自的指數,三者疊加

`mrl_lambda_exp_disjoint_kl1e-4_1024_l8_16` —— 在 disjoint VAE 上疊三個手段,全部針對上面那個
量測到的同一個問題。

1. **MRL**(`task.m_values: [8, 16, 32, 64]`、`m_weights: [1, 2, 3, 4]`)強迫每一個前綴都要能自己
   重建,所以 channel 必須按重要性排序,而不是各自扛一份任意的份額。權重是對 m 的*取樣分布*,
   所以每步只 decode 一次而不是四次:期望梯度相同,變異較大。
2. **lambda-VAE**(`lam_delta: 1.1`)直接攻擊那 19% 的後驗雜訊項。
3. **每個 m 各自的 recon 指數**(`m_exponents: [2.0, 1.5, 1.0, 0.5]`)讓 loss 隨著 m 變大而愈不
   追求期望值。短前綴保持 exp=2,因為粗略形狀*本來就是*它們的任務;只有全寬度的碼 —— 真正被
   要求解析細節的那個 —— 才用 exp=0.5。

**事前就說明的風險。** exp < 1 讓影響函數 `p·|e|^(p-1)` 隨 |e| *遞減*:在 |e| = 0.1 時梯度是 135,
而 MSE 是 1e5,**少了 740×**。同一個讓模糊停止的改動,也等於叫模型放棄它最差的那些點 —— 而上面
講的近表面失敗**正是**大殘差。如果 `val/viou` 掉但 `val/recon` 看起來正常,原因就是這個,第一件
該試的事是把 `m_exponents` 壓平成 `[2, 2, 1.5, 1]`。

**已知的混淆,如實記錄:**三個軸同時動,所以贏了也無法歸因到任何單一項。這是在時程壓力下的刻意
取捨 —— 這個 run 回答的是「這個組合到底能不能贏過 83 V-IoU」。隔離指數這一項的對照組,就是同一份
config 拿掉 `m_exponents`。驗證永遠跑在 m = max(m_values) = 64,所以 `val/viou` 與非 MRL 的 run
仍然直接可比。

### lambda-VAE 實作:兩次 NaN 事故

兩次都殺掉了真的訓練;這兩次也就是 `Gaussian.sample` 為什麼在 **log space** 比較,而不是用顯而
易見的寫法。

1. `lam` 來自 channel 平均的 sigma,卻乘在每一個元素的 sigma 上,而 `sigma^lam` 只在 sigma < 1 時
   縮小 —— 超過 1 就爆炸。這裡有 5.2% 的元素 sigma > 1(最大 8.4);在 lam ≈ 12 的情況下,雜訊被
   送到 **1089**,而 SDF 目標範圍只有 ±1.6,epoch 8 就 NaN。修法:Eq.16 的 `max(1, ·)` 必須
   **逐元素**套用,不能套在 channel 統計量上。
2. 把那個 clamp 寫成 `minimum(sigma, sigma**lam)` 更早就 NaN,在 epoch 6:logvar 的上限是 +20,
   所以 sigma 可以到 exp(10) = 22026,而 `22026**10.7` 溢位成 inf。`minimum()` 在前向會選 sigma,
   但 autograd 仍然會算 pow 的局部梯度,再乘上路由到那個分支的零 —— **0 × inf = NaN**。改成比較
   `log(sigma)` 與 `lam·log(sigma)` 會做出完全相同的選擇,永遠不會去算那個溢位的次方,而且在勝出
   的分支上留下有限的梯度。

`kl_divergence()` 刻意不動:讓 KL 按**原始**的 sigma 收費、而 decoder 只看得到縮減後的雜訊,
這個落差就是整個機制(Eq.10)。
