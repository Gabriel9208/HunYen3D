# HunYen3D

一個從零手刻的三維形狀自編碼器,架構取法 Hunyuan3D-2 —— 並診斷出它的潛在表示(latent
representation)有兩種安靜的失效方式,各自提出修正。

*[English version](README.md)*

---

## 摘要

- **發現的問題。** 在忠實復現的 Hunyuan3D-2 形狀自編碼器裡,**有一半的 latent token 完全沒有攜帶
  資訊**。而死掉的那一半,恰好是負責折角與銳利邊緣的那一半 —— 也正是人眼判斷形狀品質時最先看的
  地方。
- **成因。** 定義 latent token 位置的兩組錨點是各自獨立挑選的,因此會落在彼此身上。重複的那一組
  在訓練過程中被淘汰掉。
- **修正一:兩階段採樣。** 先挑銳邊錨點,再在銳邊錨點已經就位的前提下挑均勻錨點,讓兩組互相排斥。
  token 使用率從 54% 提升到 100%,四項重建指標全數改善。**不增加任何參數,不增加訓練時間。**
- **修正二:λ-VAE。** token 只是問題的一半;每個 token 內部的 64 個 channel 使用也不均。導入
  λ-VAE 補上第二個缺口,各項指標再次提升。
- **規模。** 3.28 億參數,在單張 RTX 5080 上以 ShapeNet-v2 訓練,並在官方測試切分的全部 2,592 個
  形狀上評估。

| 指標 | 原始採樣 | 兩階段採樣 | 兩階段 + λ-VAE |
|---|---|---|---|
| Volume IoU ↑ | 0.8980 | 0.9039 | **0.9171** |
| Chamfer Distance ↓ | 0.0142 | 0.0136 | **0.0127** |
| F-Score @ 0.02 ↑ | 0.9712 | 0.9749 | **0.9834** |
| Normal Consistency ↑ | 0.9477 | 0.9495 | **0.9547** |

*三個模型的架構與參數量完全相同,差異只在於錨點怎麼挑,以及第三欄額外改變了訓練時注入雜訊的
方式。這些數字實際量的是什麼,見[評測協定](#評測協定)。*

---

## 背景:為什麼值得研究三維生成模型的潛在空間

三維物件生成過去的做法是先生成二維影像再抬升到三維。隨著大規模三維資料集出現,這個領域轉向直接在
三維資料上訓練。目前的主流配方分成兩階段:先用自編碼器把一個三維形狀壓縮成精簡的**潛在表示**
(一小組數字,用來代表整個物件),再用擴散模型學習在那個壓縮空間裡生成新的點。

目前有兩類潛在表示在使用,取捨方向不同。

**稀疏體素方法**(XCube、TRELLIS)保留一個顯式的空間網格。每個 latent 數值都綁定在一個已知位置
上,所以生成模型只需要決定那個位置附近的表面長什麼樣 —— 表面在哪裡,網格已經先告訴它了。

**VecSet 方法**(3DShape2VecSet、Dora、Hunyuan3D)把網格丟掉。形狀變成一組沒有順序、也沒有固定
空間意義的 latent 向量。這樣壓縮率高很多,但也意味著生成模型必須用同一組數字,同時產生表面的
**位置**與**細節**。這是更難的任務,也是一般用來解釋 VecSet 方法在精細幾何上表現吃力的原因。

Hunyuan3D 團隊自己的回應是 LATTICE:把 VecSet 的輸出轉成類似體素的表示,再跑一次完整的生成流程。
這樣確實有效,但它退回了原本的前提 —— VecSet 的存在,正是為了不依賴顯式空間網格。

這引出了另一個問題,也是這個專案要問的:

> **在不重新引入顯式空間網格的前提下,能不能讓潛在表示本身的結構更好,以減輕生成模型的負擔?**

要改善一個表示,得先知道現有的這個健不健全。檢查這件事,結果才是有趣的部分。

---

## Latent 的結構長什麼樣

後面所有討論都建立在這一節上,值得花兩分鐘看。

Encoder 並不是把網格壓成一個大向量。它產生 **1,024 個 latent token**,每個 token 寬 64 個數字。
每個 token 都**錨定**在物件表面上的一個取樣點,最終描述的是那個點附近的幾何。

錨點來自兩個獨立的池,目的不同:

- **均勻池** —— 在整個表面上均勻撒點,並依三角面面積加權,以免被細分過的區域佔掉過多比例。這組
  負責整體覆蓋。
- **銳邊池** —— 沿著折角撒點。當一個頂點的法線方向與周圍面的差異超過約 10 度,就判定為折角頂點;
  一條邊要**兩端頂點都是**折角頂點,才算銳邊。再沿著這些邊撒點。這組存在的理由是:折角幾乎不佔
  表面積,均勻採樣基本上抽不到,但它們恰恰是讓一張椅子看起來像椅子的部分。

每個池各存 81,920 個點。每個訓練步驟重新抽一份子集,再用**最遠點採樣**(farthest point sampling,
一種貪婪程序,反覆挑出離已選點最遠的候選點,藉此取得分佈良好的覆蓋)把每份子集縮到 512 個錨點。
兩半接在一起構成最終的 1,024 個 token:**token 0–511 是均勻錨點,token 512–1023 是銳邊錨點。**

![採樣流程與 ShapeVAE 架構](docs/figures/architecture.svg)

*(a) 是上述的採樣流程,(b) 是完整的自編碼器。(a) 裡的小插圖標示的是較早期、小樣本的錨點間距量測;
全測試集的數字在下方表格,請以表格為準。*

---

## 發現:一半的 latent token 是空的

變分自編碼器(variational autoencoder)訓練時受到兩股壓力。一股獎勵重建輸入;另一股是
**KL 散度**(KL divergence)項,把每個 latent 數值推向一個固定的預設分布。一旦某個 token 對重建
不再有用,就沒有東西抵抗第二股壓力,它會被壓成純雜訊。這叫**後驗塌縮**(posterior collapse),
而且很容易被忽略,因為一般回報的 KL 值是整個 latent 的單一平均 —— 一半塌縮,會被另一半的活躍度
掩蓋掉。

把那個平均值按 token 拆開來看,故事完全不同。

![原始採樣的 per-token 與 per-channel KL 散度](docs/figures/vae_converge_kl1e-4_lr3.5e-6_1024_l8_16_epoch64_seed0.png)

*原始採樣。中間那張圖是每個 latent token 一根長條。均勻錨點那一半是活躍的;銳邊錨點那一半
貼在地板上。*

銳邊那一半是死的。若以 Burda 等人提出的標準 **Active Units**(活躍維度)判準來看 —— 一個維度算
「有被使用」,條件是它編碼出來的值確實會隨形狀變化,且變化量超過 0.01 門檻 —— 則錨定在銳邊上的
token 幾乎全數不合格,而錨定在均勻點上的則全部合格。

從三種粒度分別清點 —— 每一個數字、每個 channel、每個 token:

| Active Units | 全部 65,536 維 | Per-channel(64) | Per-token(1,024) |
|---|---|---|---|
| 原始採樣 | 34,911(53.3%) | 61 / 64 | 525 / 1024 |
| 兩階段採樣 | 60,617(92.5%) | 63 / 64 | **1024 / 1024** |

在測試切分的全部 2,592 個形狀上量測。

也就是說,模型付了 1,024 個 token 的代價,實際只用到大約一半,而沒用到的那一半,正是原本要負責
銳利細節的那一半。

---

## 兩個被推翻的解釋

在找到真正成因之前,先測試並排除了兩個合理的假設。兩者在這個 repo 裡都還是可執行的程式碼。

**會不會是 token 分辨不出彼此?** Encoder 的注意力機制本身不知道每個錨點相對於其他錨點在哪裡,
所以銳邊 token 塌縮,或許只是因為模型無法區分它們。若是如此,加入 **旋轉位置編碼**(rotary
position embedding,一種把相對位置資訊注入注意力的標準做法)應該就能解決。它確實改變了模型的
行為 —— 注意力變得明顯更局部 —— 但塌縮照樣發生,重建品質也沒有提升。位置資訊不足並不是成因。
*實作:*`src/model/shape/VAE/vae.py` 的 `RoPEVAE`。

**會不會是均勻錨點根本不必要?** 人只看到物體的邊緣,通常就能推斷出完整形狀。如果模型也適用同樣的
道理,那麼只用銳邊錨點就夠了,均勻分支塌不塌縮也就無關緊要。把 query 完全換成銳邊點,可以直接檢驗
這一點。結果重建品質**反而變差**,說明均勻錨點確實攜帶了必要的監督訊號,不能捨棄。
*實作:*`src/model/shape/VAE/vae.py` 的 `FrameVAE`。

---

## 成因:兩組錨點疊在一起

把錨點畫在三維空間裡之後就一目了然。兩個池是各自獨立採樣的,而最遠點採樣只會把點在**自己的池
裡面**撐開。沒有任何機制阻止一個均勻錨點落在幾乎與某個銳邊錨點相同的位置。

![原始採樣與兩階段採樣的錨點位置](docs/figures/anchor_overlap_chair.png)

*上排:原始採樣。下排:兩階段採樣。左、中兩欄分別是單獨一個分支;右欄把兩者疊起來,黑色圓圈標出
距離小於 0.02 的配對。在這張椅子上,原始方法產生 46 對碰撞(共 512 個),最近配對距離 0.0013;
兩階段方法沒有任何碰撞,最近配對距離 0.0712。下方直方圖對所有錨點呈現同一件事:兩個分布幾乎沒有
重疊。*

放到全測試集上看,這個現象是系統性的。一個形狀裡最接近的一對錨點平均距離是 **0.0022** —— 在整個
物件跨度為 2.0 單位的座標系裡,這等於兩個 token 在描述同一個點。

這就解釋了塌縮。當兩個 token 看到幾乎相同的鄰域時,其中一個就是多餘的,而 KL 項恰恰就是一股淘汰
冗餘的壓力。銳邊錨點在這場競爭中落敗,有兩個原因:均勻分支覆蓋的表面範圍更廣,在定義於整個空間的
重建損失下更廣泛地有用;而且每個銳邊點都帶著一個明確的旗標標示自己是銳邊點,這讓模型很容易辨識出
重複的那一組並把它丟掉。

---

## 修正一:兩階段(有種子的)最遠點採樣

修正方式直接從成因推導而來,而且改動很小。

不要讓兩次挑選各自獨立,改成**有順序地**進行:

1. 先用最遠點採樣在銳邊池上挑出銳邊錨點。
2. 再挑均勻錨點 —— 但開始時就把銳邊錨點視為「已經選過」。

因為最遠點採樣永遠挑離已選點最遠的候選,第 2 步會自動避開銳邊錨點已經佔據的區域。兩組錨點因此形成
互補而非重疊。其他一切不變:相同架構、相同參數量、相同 token 數、相同訓練成本。

| 錨點間距(全部 2,592 個測試形狀的平均) | 原始 | 兩階段 | 倍數 |
|---|---|---|---|
| 最接近的一對錨點(不分分支) | 0.0022 | **0.0517** | 23× |
| 每個均勻錨點到最近銳邊錨點的距離 | 0.0022 | **0.0553** | 25× |
| 所有錨點的平均間距 | 0.0448 | 0.0621 | 1.4× |
| 僅銳邊錨點之間的平均間距 | 0.0696 | 0.0696 | 不變 |
| 僅均勻錨點之間的平均間距 | 0.0867 | 0.0681 | 0.8× |

第四列是關鍵的對照:銳邊錨點自己的間距在兩種方法下**完全相同**。這個改動只作用在兩組錨點**之間**
的關係,而這正是它被設計來做的事。

塌縮消失了:

![兩階段採樣的 per-token 與 per-channel KL 散度](docs/figures/kl_disjoint.png)

*兩階段採樣,版面與 token 排列順序與前一張完全相同。1,024 個 token 全數位於門檻之上,而且銳邊那半
(512–1023)反而成為比較活躍的一邊。*

在全測試集上,**1,024 個 token 全部都在使用中**,而且銳邊錨點那一半從被忽略的一邊,變成比較活躍
的一邊。

---

## 修正二:用 λ-VAE 處理 channel 這個軸

修好 token 之後,下一個問題浮現出來。所有 token 都在用了,但每個 token 內部的 **64 個 channel**
仍然沒有平均分擔工作。即使修好取樣之後,最忙的 channel 隨形狀變化的幅度仍是最閒的 8.5 倍 ——
在修正之前是 10.7 倍,可見兩階段採樣幾乎沒有動到這個軸。

要理解為什麼這是問題,得回想 encoder 實際上輸出了什麼。對每一個 latent 數字,它輸出兩樣東西:一個
值,以及一個不確定度。訓練時模型會加入以該不確定度為尺度的隨機雜訊,然後才把 latent 交給 decoder
—— 這個雜訊正是迫使潛在空間變得平滑、而不是變成查表的關鍵。但如果某個 channel 的訊號很小而雜訊
不小,decoder 根本聽不見它,也就永遠學不會使用那個 channel。

**λ-VAE** 用一個改動加上一個刻意的「不改」來處理這件事:

- **壓低 decoder 實際收到的雜訊。** 在用不確定度來縮放雜訊之前,先把每個 channel 的不確定度取一個
  大於一的次方。由於這些不確定度都小於一,取次方會讓它們變小 —— 而且原本越小的,被壓得越多。因此
  弱 channel 得到最大的紓解,訊噪比在 64 個 channel 之間被拉平。
- **KL 罰則原封不動。** 罰則仍然是用 encoder 的**原始**不確定度計算,而不是壓低後的那個。這是讓
  這個方法站得住腳的部分:decoder 收到更乾淨的訊號,但模型仍然被收取完整的正則化代價,潛在空間
  不會被偷偷放寬。

![各 channel 的訊號、雜訊與訊噪比](docs/figures/per_channel_usage.png)

*左:各 channel 的值隨形狀變化的幅度,已排序。中:各 channel 收到的雜訊量。右:兩者的比值。
λ-VAE 把各 channel 的雜訊拉平(中圖綠線),使訊噪比中位數相對兩階段採樣提升約一倍,從 0.32 到
0.73(原始採樣為 0.23)。*

channel 之間的不均從 8.5 倍壓到 3.2 倍,重建指標再次全數提升:

| 指標 | 原始採樣 | 兩階段採樣 | 兩階段 + λ-VAE |
|---|---|---|---|
| Volume IoU ↑ | 0.8980 | 0.9039 | **0.9171** |
| Chamfer Distance ↓ | 0.0142 | 0.0136 | **0.0127** |
| F-Score @ 0.02 ↑ | 0.9712 | 0.9749 | **0.9834** |
| Normal Consistency ↑ | 0.9477 | 0.9495 | **0.9547** |

| Active Units | 全部 65,536 維 | Per-channel(64) | Per-token(1,024) |
|---|---|---|---|
| 原始採樣 | 34,911(53.3%) | 61 / 64 | 525 / 1024 |
| 兩階段採樣 | 60,617(92.5%) | 63 / 64 | 1024 / 1024 |
| 兩階段 + λ-VAE | **65,536(100%)** | **64 / 64** | **1024 / 1024** |

Active Units 在這裡已經飽和在 100%,無法再顯示 λ-VAE 多走了多遠。承擔這件事的是上面的 channel
數據:channel 之間的不均從 8.5 倍降到 3.2 倍,訊噪比中位數從 0.32 升到 0.73,這兩點都是 token 層級
的計數表達不了的。

兩個修正作用在不同的軸上 —— 一個管 token,一個管 channel —— 所以可以疊加,而且兩者都沒有增加
任何一個參數。

---

## 評測協定

這個領域的重建數字很容易在不經意間被灌水,所以有必要把量的是什麼講清楚。

**資料。** ShapeNet-v2,使用 3DILG 作者發布的 watertight 網格與 train/validation/test 切分。上述
所有數字都是完整測試切分:**2,592 個形狀**,每個模型都在相同的形狀、相同的查詢點上評估。

**四個指標的白話定義。**

- **Volume IoU** —— 對一組探測點,模型與真實答案對於「哪些點在物件內部」是否一致?回報的是兩個
  「內部點集合」的重疊比例。這衡量的是整體體積的正確性。
- **Chamfer Distance** —— 在重建表面與真實表面上各取一組點;計算每個點到對方集合中最近點的距離,
  雙向平均。這衡量幾何誤差的絕對大小,因此越低越好。
- **F-Score** —— 在重建表面點中,有多少比例落在真實表面的容忍距離內,反方向亦然,再合併為單一
  分數。與 Chamfer 不同,它不會被少數大離群值主導。
- **Normal Consistency** —— 兩個表面上的對應點,**朝向**是否一致?這能抓到以位置為準的指標抓不到
  的方向性錯誤。

Volume IoU 在單位立方體內均勻取樣的 50,000 個查詢點上計算;Chamfer Distance 與 F-Score 則從重建
網格與真實網格各取樣 100,000 個表面點後計算,其中重建網格以 marching cubes 在 128³ 網格上抽取。
F-Score 的容忍距離設為 0.02,以對應 marching cubes 在該解析度下能解析的點距。

完整結果表:[`docs/writeup/eval_result_lambda.md`](docs/writeup/eval_result_lambda.md)。

---

## 這個改善是真的嗎?

絕對差距很小 —— Chamfer 改善 4.5%,Volume IoU 大約半個百分點。平均值的小差異值得懷疑,所以另外
做了**逐形狀**比較。由於每個模型看到的都是完全相同的 2,592 個形狀,每一個形狀都可以直接判定勝負。

| 指標 | 兩階段勝出的形狀比例 | Wilcoxon 符號秩檢定 |
|---|---|---|
| Volume IoU | 70.4% | p < 1e-100 |
| Surface IoU | 70.7% | p < 1e-130 |
| Chamfer Distance | 76.1% | p < 1e-170 |
| F-Score | 66.0% | p < 1e-95 |
| Normal Consistency | 71.3% | p < 1e-115 |

改善幅度小但高度一致,而後者才是比較有用的性質。以 F-Score 為例,兩階段大幅勝出的形狀數量約為
大幅落後的五倍 —— 這個提升是散布在整個資料集上的,而不是由少數離群值撐起來的。

改善出現的位置也符合方法的預測。Surface IoU 只計算表面附近薄殼內的點,提升 1.21%,高於 Volume IoU
的 0.66%;而 Chamfer —— 四者中最直接衡量幾何的 —— 改善幅度最大。效果作用在**表面精度**上,這正是
把同樣數量的錨點鋪到更大表面範圍所應該得到的結果。

---

## 模型與訓練設定

架構沿用 Hunyuan3D-2 的形狀自編碼器。Encoder 從 1,024 個錨點 query 對一組更大的表面點做
cross-attention,經過 8 層 self-attention,為每個 token 的 64 個 channel 各輸出一個值與一個不確定
度。Decoder 把取樣後的 latent 送過 16 層 self-attention,再從空間中任意查詢位置做 cross-attention,
預測**有號距離函數**(signed distance function)—— 對空間中任一點,它離表面多遠,內部為負、外部
為正。重建損失是對真實距離值的均方誤差。

見上方架構圖的 (b) 部分。

| | |
|---|---|
| 參數量 | 327,824,001(三個模型完全相同) |
| Latent | 1,024 個 token × 64 個 channel |
| 寬度 / 頭數 | 1,024 / 16 |
| 層數 | Encoder 8 層,Decoder 16 層 |
| Encoder 輸入 | 1,024 個 query,20,480 個 key/value 點 |
| 每點特徵 | 座標 (3) + 法向量 (3) + 銳邊旗標 (1) |
| 監督 | 每步 16,384 個距離取樣點,取自 250,000 點的點庫 |
| 學習率 | 3.5e-5 訓練至收斂,再以 3.5e-6 訓練至收斂 |
| KL 權重 | 1e-4 |
| Batch | 2,梯度累積 16 |
| 精度 | bfloat16 autocast |
| 硬體 | 單張 RTX 5080(16 GB) |

---

## 其他探索過的方向

上面三個模型是有完整測試集數字的。另外還建了並跑過一些想法,在此列出以求完整,並如實標示各自的
狀態而非硬湊結論。多數在證據指向別處之後就停掉了。

| 變體 | 它在問什麼 | 結果 |
|---|---|---|
| `RoPEVAE` | 塌縮是不是因為 token 缺少相對位置資訊? | 不是。注意力局部性從 53.4% 升到 77.4%,但重建沒有改善,塌縮照樣發生。即上文兩個被推翻的假設之一。 |
| `FrameVAE` | 均勻錨點是不是多餘的 —— 銳邊錨點能不能獨力完成? | 不能。重建變差,均勻分支攜帶了必要訊號。另一個被推翻的假設。 |
| `DoubleStreamVAE` | 如果均勻錨點與銳邊錨點其實是兩種性質不同的輸入,把它們當成兩個 modality 會不會有幫助 —— 就像 MM-DiT 讓文字與影像各走一條 stream、再用 joint attention 交換資訊? | 重建結果差不多,但 encoder 的參數量翻倍(113.6M → 226.8M),因為每條 stream 各自帶一組投影、cross-attention 與 MLP。同樣的結果花兩倍代價,所以沒有採用。 |
| `MaskedVAE` | 若均勻 token 可以看全部、銳邊 token 只能看銳邊點,銳邊分支會不會就不再被吸收掉? | 以相同層數建構,不增加容量。收斂速度明顯更快,但從 channel 的角度看,死掉的 channel 非常多,因此正在疊上 λ-VAE 來改善。實驗仍在進行中。 |
| `MRLVAE` | 能不能讓 64 個 channel 按重要性排序,使 latent 的任何前綴都能單獨使用? | 以逐前綴的重建損失與逐前綴的損失指數訓練。屬探索性質,沒有收斂的測試集數字。 |
| 根號重建損失 | 它有助於細節的 refinement 嗎?相對於均方誤差,根號損失把梯度從殘差最大的點重新分配到殘差最小的點。在 eps=1e-4 下實測每點梯度:殘差 1e-4 處是 MSE 的 81 倍,1e-1 處只有 MSE 的 1/233,兩者在 4e-3 附近相當 —— 那大致就是目前的平均誤差。而細節是在小殘差那一端被決定的。 | 拿 MSE 預訓練好的模型接著用根號損失續訓,確實看得到效果。但這是尚未收斂的 run 上的初步觀察。實驗仍在進行中。 |

這個 repo 裡也實作了生成模型 —— 以 rectified flow 為訓練目標的 diffusion transformer —— 但尚未在
改進後的 latent 上訓練。因此,更好的 latent 結構是否真的有助於生成,目前仍是開放問題,不是這裡所
主張的結論。

---

## 專案結構

```
main.py                       Hydra 進入點 → src.engine.runner.run
configs/                      Hydra config 群組
  config.yaml                 頂層預設
  model/ task/ data/ loss/    架構、目標函數、資料集、重建損失
  capacity/                   各 experiment 共用的 latent 數量 / 深度預設
  experiment/                 一個檔對應一次 run,以 +experiment=<name> 啟用
src/
  model/shape/VAE/            encoder、decoder、各 VAE 變體
  model/shape/preprocess.py   網格 → 錨點 + 距離監督(兩種採樣法都在這裡)
  model/shape/diffusion/      MM-DiT 去噪器與 flow matching
  engine/                     Trainer、task 定義、資料、checkpoint、logging
scripts/
  build_cache.py              離線前處理快取(訓練前先跑)
  paper_eval.py               多個 checkpoint 並排評測
  evaluate.py                 單一 run 的診斷評測
  kl_histogram.py             per-token / per-channel KL 與 Active Units
  anchor_scatter.py           錨點重疊的三維圖
  smoke_test.py               跑幾步真訓練,確認記憶體放得下
docs/
  figures/                    本 README 使用的圖
  writeup/
    eval_result_lambda.md     主結果表
    eval_result_256.md        解析度穩健性檢查
```

---

## 開始使用

相依套件以 [uv](https://docs.astral.sh/uv/) 管理(Python ≥ 3.13):

```bash
uv sync
```

把 watertight 網格放到 `data/train`、`data/val` 與 `data/test`(皆已 gitignore)。

前處理分成兩部分。昂貴且確定性的那一半 —— 把網格轉成距離場、建立兩個點池 —— 離線做一次並快取。
便宜且隨機的那一半 —— 抽子集、選錨點、算位置特徵 —— 每個 epoch 都重跑,讓每個 epoch 看到新的取樣。

```bash
# 1. 建快取(只需一次)
uv run python scripts/build_cache.py +experiment=vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16

# 2. 確認幾步真訓練放得進記憶體
uv run python scripts/smoke_test.py +experiment=vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16

# 3. 訓練
uv run python main.py +experiment=vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16
```

結果表中的三個模型對應以下 experiment config:

| 表格欄位 | Experiment config |
|---|---|
| 原始採樣 | `vae_converge_kl1e-4_lr3.5e-6_1024_l8_16` |
| 兩階段採樣 | `vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16` |
| 兩階段 + λ-VAE | `vae_lambda_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16` |

任何欄位都可以從命令列覆寫,長時間的訓練也可以續跑:

```bash
uv run python main.py +experiment=<name> optimizer.lr=1e-5 trainer.max_epochs=40
uv run python main.py +experiment=<name> trainer.resume=/abs/path/to/last.pt
```

要重現跨 checkpoint 的比較表:

```bash
uv run python -m scripts.paper_eval \
    --ckpt vanilla=/abs/path/best.pt \
    --ckpt disjoint=/abs/path/best.pt \
    --ckpt lambda=/abs/path/best.pt \
    --shapes 0 --device cuda --md-out docs/writeup/eval_result.md
```

---

## 限制與後續

**兩階段採樣犧牲了一部分隨機性。** 原始方法的兩組錨點是獨立抽取的;兩階段方法的第二次抽取受第一次
約束,兩者不再獨立。在訓練 epoch 數不多時,這個代價量不出來,但若長時間訓練而候選池又不夠大,
模型可能會反覆看到相同的配對,降低實際的資料多樣性。擴大採樣池,或設計一個保留更多隨機性的耦合
採樣器,是明顯的下一步。

**這個方法針對的是一種特定的浪費。** 它回收的是折角錨點與均勻錨點重疊所損失的容量。在銳利特徵稀少
的資料上 —— 有機形體、掃描模型 —— 兩組錨點本來就不容易重疊,效益應該會相應縮小。

**生成尚未驗證。** 這裡量到的全部是重建品質與 latent 結構。至於「更好的 latent 結構能讓下游生成
模型更輕鬆」這個前提,在這個 repo 裡還沒有被檢驗。

---

## 參考文獻

**本專案復現的對象**

- Tencent Hunyuan3D Team. *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D
  Assets Generation.* arXiv:2501.12202. —— 本專案重新實作的形狀自編碼器與採樣流程。
- Tencent Hunyuan3D Team. *Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with
  Production-Ready PBR Material.* arXiv:2506.15442. —— 撰寫實作時參照的開源釋出版本,也是交叉檢查
  所使用的官方 checkpoint 來源。
- Tencent Hunyuan3D Team. *Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate
  Details.* arXiv:2506.16504. —— 提出 LATTICE,即把 VecSet 輸出轉成類似體素的表示、再跑第二次生成
  的形狀基礎模型。在背景一節中作為本專案刻意不採取的替代路線引用。

**三維生成的潛在表示**

- Biao Zhang, Jiapeng Tang, Matthias Nießner, Peter Wonka. *3DShape2VecSet: A 3D Shape
  Representation for Neural Fields and Generative Diffusion Models.* SIGGRAPH 2023.
  arXiv:2301.11445. —— VecSet 表示法,以及本文沿用的評測慣例(均勻查詢點、逐形狀平均)。
- Rui Chen et al. *Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders.*
  CVPR 2025. arXiv:2412.17808. —— 形狀自編碼器的銳邊取樣,以及針對細節豐富點雲的雙 cross-attention
  encoder。
- Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, Francis Williams. *XCube:
  Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies.* CVPR 2024(Highlight).
  arXiv:2312.03806. —— 稀疏體素潛在表示,在背景一節作為另一條路線引用。
- *Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS).* CVPR 2025(Spotlight).
  arXiv:2412.01506. —— 背景一節引用的另一個稀疏體素代表。

**資料**

- Biao Zhang, Matthias Nießner, Peter Wonka. *3DILG: Irregular Latent Grids for 3D Generative
  Modeling.* NeurIPS 2022. arXiv:2205.13914. —— watertight ShapeNet-v2 網格,以及本文所有數字所
  使用的 train/validation/test 切分的來源。

**使用到的方法與判準**

- Girum Demisse. *λ-VAE: Variance Equalization for Posterior Collapse.* arXiv:2607.05531. ——
  修正二的依據:以逐維度的指數縮放取樣雜訊,而 KL 罰則仍按原始的後驗變異數計算。
- Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. *Importance Weighted Autoencoders.* ICLR 2016.
  arXiv:1509.00519. —— 全文使用的 Active Units 判準。
- Patrick Esser, Sumith Kulal, Andreas Blattmann et al. *Scaling Rectified Flow Transformers for
  High-Resolution Image Synthesis.* arXiv:2403.03206. —— 啟發 `DoubleStreamVAE` 的 MM-DiT 雙流
  區塊,也是本 repo 中 diffusion transformer 所用的 rectified flow 訓練目標。
- Aditya Kusupati, Gantavya Bhatt, Aniket Rege et al. *Matryoshka Representation Learning.*
  arXiv:2205.13147. —— `MRLVAE` 背後的前綴排序構想。
- Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
  arXiv:2104.09864. —— 在 `RoPEVAE` 中測試的旋轉位置編碼。
- William E. Lorensen, Harvey E. Cline. *Marching Cubes: A High Resolution 3D Surface Construction
  Algorithm.* SIGGRAPH 1987. —— Chamfer、F-Score 與 Normal Consistency 所使用的表面抽取演算法。


讀過什麼、試過什麼、量到什麼 —— 包含失敗的嘗試 —— 的研究日誌保留在本機、未隨 repo 發布,
因此各項決定的理由都直接寫在上面對應的章節裡。
