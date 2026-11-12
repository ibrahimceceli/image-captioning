# Model 2 beam-3: 40 görüntülük görsel inceleme

Bu bir val örneklemi; test görüntüsü kullanılmadı. Tüm görüntüler, tahminler ve beşer referans ilk görsel incelemede birlikte değerlendirildi. İşaretlenen kayıtlar kullanıcı kararlarıyla sonuçlandırılır; etiketler bağımsız ground truth değildir.

Seçim: 1.000 val ID üzerinde `SHA256("model2-beam3-review-20261112-v1:" + image_id)` artan sırası, ilk 40. Kalite filtresi yok. Seçim manifesti `selection.json`.

Fotoğrafları açmak için yerel Flickr8k kurulumu gerekir; görüntü kimlikleri `data/flickr8k/Images/` altında aranır. Veri seti teslim paketine dahil değildir.

| Kategori | Kesin sayı / 40 | Örnek görüntü ID | Kısa gerekçe |
|---|---:|---|---|
| yanlış nesne | 11 | 3168354472_866fe70d36.jpg | Çocuk yerde uzanıp ağlıyor; bed/jumping ve kırmızı üst yerine mor elbise görünüyor. |
| sayı | 0 | — | Bu kategoride kesin hata etiketi yok. |
| renk | 10 | 3168354472_866fe70d36.jpg | Çocuk yerde uzanıp ağlıyor; bed/jumping ve kırmızı üst yerine mor elbise görünüyor. |
| eylem | 11 | 3168354472_866fe70d36.jpg | Çocuk yerde uzanıp ağlıyor; bed/jumping ve kırmızı üst yerine mor elbise görünüyor. |
| genel ifade | 18 | 3249014584_21dd9ddd9d.jpg | Üst giysi siyah, mavi değil. Ayakta oluş doğru; buz blokunu testereyle yontma ana eylemi ve alet atlanmış. |
| kelime tekrarı | 4 | 3553225222_f5ebe44af1.jpg | Çocuk kırmızı mezuniyet cübbesiyle kep atıyor; blue shirt iki kez yineleniyor ve mezuniyet/eylem yok. |

## Kodlama kuralları

- **yanlış nesne:** Görselde olmayan nesne/giysi/ortam eklenmesi veya nesnenin yanlış adlandırılması. Salt nesne atlama tek başına bu etiketi almaz. Cinsiyet görüntüden varsayılmaz.
- **sayı:** Açıkça yanlış adet belirtilmesi. Tekil bir aktörü anlatıp diğerini atlamak tek başına sayı hatası sayılmaz; bağlam kaybı varsa genel ifade olarak değerlendirilir.
- **renk:** Mevcut nesne/giysinin açıkça yanlış rengi. Renk atlama hata sayılmaz; olmayan giysi için renk yerine yanlış nesne kullanılır.
- **eylem:** Ana eylem veya mekânsal ilişkinin görünür durumla çelişmesi. Yalnız eylem atlama genel ifade altında değerlendirilir.
- **genel ifade:** Sahnenin ayırt edici ana bağlamını veya aktörler arası etkileşimi atlayan açıklama. Ana etkinliği doğru veren kısa açıklamalarda salt ayrıntı eksikliği hata sayılmaz; yarım kalma tek başına bu etiketi zorunlu kılmaz.
- **kelime tekrarı:** Gereksiz aynı içerik sözcüğü/öbeği tekrarı; sıradan a/the/is ve zorunlu sözdizimsel tekrarlar sayılmaz.

Başlangıçta kullanıcı kontrolü bekleyen **18/18 kayıt tamamlandı**; bekleyen kayıt yok. 40 görüntünün **28** tanesinde en az bir kesin kategori etiketi, **12** tanesinde bu altı kategoride kesin etiket yok. **6 görüntüde belirsiz etiket** ayrı tutuldu; bunlar kesin kategori toplamlarına dahil edilmedi. Toplam **54 kategori ataması** var. Bir görüntü birden fazla kategoriye girebildiği için kategori sayıları örtüşür; toplamları farklı hatalı görüntü sayısı değildir. İnsan kontrolüne işaretlenmeyen kayıtların ilk görsel etiketleri korundu; sonuçlar bu 40 görüntüyü betimler, tüm val kümesine genellenmez.

### Belirsiz kayıtlar — kesin toplamlardan hariç

| Kayıt | Görüntü ID | Belirsiz etiket |
|---|---|---|
| 03 | 1865794069_6e3a1e57bb.jpg | belirsiz — sayı |
| 08 | 319847643_df7c2a1d25.jpg | belirsiz — eylem |
| 18 | 3265864834_e0229020dd.jpg | belirsiz — eylem |
| 23 | 3220126881_b0a4f7cccb.jpg | belirsiz — eylem |
| 38 | 3052038928_9f53aa2084.jpg | belirsiz — yanlış nesne |
| 40 | 2599903773_0f724d8f63.jpg | belirsiz — kişi niteliği (boy/girl) |

Nihai karar kaynağı `annotations.json`; `review40.json` görüntü, tahmin ve referansları da saklar. CSV’de `human_labels`, `human_notes`, `uncertain_labels` ve `review_status` insan kararlarını korur.

## 01 — 3168354472_866fe70d36.jpg

**Görüntü kimliği:** `3168354472_866fe70d36.jpg`

**Beam-3:** a little boy in a red shirt is jumping on a bed

**Etiketler:** yanlış nesne, renk, eylem

**Gerekçe:** Çocuk yerde uzanıp ağlıyor; bed/jumping ve kırmızı üst yerine mor elbise görünüyor.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Not / belirsizlik:** Boy/girl ayrımı ayrıca etiketlenmedi; üç etiket yatak, renk ve eylem kanıtına dayanıyor.

**Beş referans:**

- A girl in a purple dress crying on the floor .
- A girl in a purple dress is throwing a tantrum .
- A little girl in a purple plaid dress is lying on the floor crying .
- A little girl in a purple plaid dress lays on the floor and cries with her toy beside her .
- A young girl in a purple dress throws a tantrum .

## 02 — 3249014584_21dd9ddd9d.jpg

**Görüntü kimliği:** `3249014584_21dd9ddd9d.jpg`

**Beam-3:** a man in a blue jacket is standing in the snow

**Etiketler:** renk, genel ifade

**Gerekçe:** Üst giysi siyah, mavi değil. Ayakta oluş doğru; buz blokunu testereyle yontma ana eylemi ve alet atlanmış.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- a man carves an ice block with a chainsaw .
- A man has a chainsaw and is carving a block of ice .
- A man in black shaving a ice sculpture .
- Man carves block of ice with a saw .
- Man with an ice sculpture

## 03 — 1865794069_6e3a1e57bb.jpg

**Görüntü kimliği:** `1865794069_6e3a1e57bb.jpg`

**Beam-3:** two boys play soccer

**Etiketler:** belirsiz — sayı

**Gerekçe:** Bir çocuğun futbol oynadığı açık, ikinci çocuğun oyuna katıldığı doğrulanamadı

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Not / belirsizlik:** Bir çocuğun futbol oynadığı açık, ikinci çocuğun oyuna katıldığı doğrulanamadı

**Beş referans:**

- A blond boy with a navy blue uniform kicks a soccer ball in an open field .
- A boy in an all blue uniform kicks a blue soccer ball in a grassy area .
- A little boy is playing soccer at a field .
- A small child plays with a ball while others relax and converse with one another .
- A young boy dressed in blue and white is kicking a soccer ball .

## 04 — 2635023078_6dae04758f.jpg

**Görüntü kimliği:** `2635023078_6dae04758f.jpg`

**Beam-3:** a man and a man in a black leather jacket is standing on a sidewalk

**Etiketler:** Bu altı kategoride kesin etiket yok

**Gerekçe:** Man sözcüğü iki ayrı kişiye gönderme yapıyor. Cümle doğal değil ve çoğul özneyle ‘are’ kullanılmalı

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- Two guys stand outside of a building that has been spray painted with words .
- Two men stand near a wall that has writing painted on it .
- Two young men drink beer , leaning on a graffitied wall .
- Two young men in black jackets hanging out by a graffitied wall
- Two young men in jeans and dark jackets standing in front of a building .

## 05 — 2533642917_a5eace85e6.jpg

**Görüntü kimliği:** `2533642917_a5eace85e6.jpg`

**Beam-3:** a girl in a pink bathing suit is swimming

**Etiketler:** renk, eylem

**Gerekçe:** Çocuk sığ suda ayakta; swimming değil. Görünen üst gri/mavi, şort renkli; pink bathing suit görünmüyor.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Not / belirsizlik:** Boy/girl referans farkı bağımsız görsel cinsiyet etiketi olarak kullanılmadı.

**Beş referans:**

- A boy in a bathing suit stands in water .
- A little boy stands in a creek on a sunny day .
- a toddler wearing colorful shorts is standing in the water .
- A wet child standing in water .
- Little boy knee-deep in water .

## 06 — 2085078076_b9db242d21.jpg

**Görüntü kimliği:** `2085078076_b9db242d21.jpg`

**Beam-3:** a little boy is playing in the sand

**Etiketler:** genel ifade

**Gerekçe:** Tahmin, sahneyi ayırt eden su kenarını ve kovaları belirtmiyor. Kumda oynama ifadesi doğrulanamadığı için kesin eylem hatası olarak sayılmadı

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- A small boy standing near muddy water with something in his hand while someone beside him washes clothing in buckets .
- A young black child stands on the edge of a body of water near buckets .
- A young boy stands near the water in a muddy river .
- Child standing by a rivers edge with laundry in buckets next to them .
- On shore near brown water , boy holds object beside buckets of rags .

## 07 — 3275711232_e261143664.jpg

**Görüntü kimliği:** `3275711232_e261143664.jpg`

**Beam-3:** a woman in a black coat and sunglasses

**Etiketler:** yanlış nesne, genel ifade

**Gerekçe:** Güneş gözlüğü görünmüyor; gözler açıkta ve atkı yüzü örtüyor. Yüzü örtme ana eylemi atlanmış.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A person in a knit hat hiding their face with a scarf .
- A person in winter clothes covers their face with a sweater .
- A woman in a park wearing a hat and holding a scarf over her face .
- A woman outside covers her face with fabric .
- The adult is covering their face with a piece of clothing outside

## 08 — 319847643_df7c2a1d25.jpg

**Görüntü kimliği:** `319847643_df7c2a1d25.jpg`

**Beam-3:** a little girl in a yellow shirt is blowing a

**Etiketler:** belirsiz — eylem

**Gerekçe:** Tahmin ‘blowing a’ ile yarım kalıyor. Üfleme eylemi ve giysi rengi doğrulanamadı

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Not / belirsizlik:** Tahmin ‘blowing a’ ile yarım kalıyor. Üfleme eylemi ve giysi rengi doğrulanamadı

**Beş referans:**

- A child pushes its face up towards a window of a playground .
- A girl cringes when drops of water come her way .
- Girls squints , presses against oval hole in wet , colorful structure .
- One boy who has a funny expression on his face has his hands up by his head and is looking through a hole .
- Water is spraying on the child with its face through a cut out hole .

## 09 — 241345596_91e0e2daf5.jpg

**Görüntü kimliği:** `241345596_91e0e2daf5.jpg`

**Beam-3:** a football player in a red jersey

**Etiketler:** genel ifade

**Gerekçe:** Kırmızı formalı oyuncu doğru; birden çok oyuncunun etkileşimi ve tribün bağlamı atlanmış. Tek bir oyuncuyu tanımlamak tek başına yanlış sayı sayılmadı.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- a crowd wearing red , cheers on the red football team .
- Football players in red congratulate each other as crowds in red cheer behind .
- The Oklahoma Sooners football team discuss their game while fans cheer .
- Two football players talk during a game .
- Two Oklahoma Sooner football players talk on the sideline .

## 10 — 2140747429_62cfd89ae9.jpg

**Görüntü kimliği:** `2140747429_62cfd89ae9.jpg`

**Beam-3:** a young boy in a black and white jacket

**Etiketler:** genel ifade

**Gerekçe:** Siyah ve beyaz alanlar görüldüğü için kesin renk hatası sayılmadı. Tahmin karı ve kar savurma eylemini belirtmiyor

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- A child in the snow with a house in the background .
- A little boy is throwing snow around outside .
- Snow covers the ground and a young boy 's jacket .
- The little boy with the yellow stripe on his hat is in the snow .
- This boy is throwing a snowball .

## 11 — 3099923914_fd450f6d51.jpg

**Görüntü kimliği:** `3099923914_fd450f6d51.jpg`

**Beam-3:** a group of people gather

**Etiketler:** genel ifade

**Gerekçe:** Açık hava kafe masasında oturup içecek içen insanlar; yalnız group of people gather ayırt edici bağlamı vermiyor.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A group of people sit at a table in front of a large building .
- People are drinking and walking in front of a brick building .
- People are enjoying drinks at a table outside a large brick building .
- Two people are seated at a table with drinks
- Two people are sitting at an outdoor cafe in front of an old building .

## 12 — 3606846822_28c40b933a.jpg

**Görüntü kimliği:** `3606846822_28c40b933a.jpg`

**Beam-3:** a man in a red shirt is jumping off a

**Etiketler:** renk, genel ifade

**Gerekçe:** Kaykaycının üstü yeşil, kırmızı değil; jumping off a tamamlanmamış ve kaykay nesnesini atlıyor.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A man does a jumping trick on a skateboard .
- A man in a green shirt on a half pipe , doing a skateboard stunt .
- A man on a skateboard while the crowd watches .
- A skateboard is sideways in the air .
- a skateboarder wearing a green shirt and jumping in the air

## 13 — 432490118_54a9c0e500.jpg

**Görüntü kimliği:** `432490118_54a9c0e500.jpg`

**Beam-3:** a brown and white dog is running through a field

**Etiketler:** eylem, genel ifade

**Gerekçe:** Tahmin ikinci köpeği ve aralarındaki etkileşimi atlıyor. Büyük köpeğin diğerine doğru sıçramasını arazide koşma olarak anlatıyor

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- A Collie and a Springer Spaniel playing outside .
- The large brown and white dog is jumping towards the small brown and white dog .
- Two brown and white dogs fighting on a grassy area in front of a tree .
- Two brown and white dogs in the grass near a stone structure .
- Two tan and white dogs look at each other in an outdoor setting .

## 14 — 270816949_ffad112278.jpg

**Görüntü kimliği:** `270816949_ffad112278.jpg`

**Beam-3:** a man stands on a mountaintop

**Etiketler:** eylem

**Gerekçe:** İnsan dağ zirvesinde ayakta değil; ön plandaki düz/çayırlık alanda at üstünde. Atın yalnız atlanmasını yanlış nesne diye saymadım.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A man on horseback on a grassy field in front of mountains .
- A person on a brown horse in a prairie in front of tall mountains .
- A person riding a horse with a mountain range in the far background .
- A rider on a horse in front of a mountain range
- Distant horse rider on plain in the middle of brown mountains .

## 15 — 512045825_1be2083922.jpg

**Görüntü kimliği:** `512045825_1be2083922.jpg`

**Beam-3:** a black and white dog is jumping over a hurdle

**Etiketler:** Bu altı kategoride kesin etiket yok

**Gerekçe:** Siyah-beyaz köpek bariyer/çit üzerinden atlıyor; tahmin ana içeriğe uyuyor.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A dog jumping over a fence .
- A dog jumps over a barrier in a field .
- A white dog with black spots is jumping over a white fence .
- The dog leaps over the fence in the park .
- The spotted dog jumps a fence .

## 16 — 3257207516_9d2bc0ea04.jpg

**Görüntü kimliği:** `3257207516_9d2bc0ea04.jpg`

**Beam-3:** a woman in a blue shirt with a black

**Etiketler:** genel ifade

**Gerekçe:** Tahmin ikinci kişiyi ve birlikte yaptıkları barış işaretini atlıyor. Cümle ‘with a black’ ifadesiyle yarım kalıyor. Giysi türü ve rengi için kesin hata etiketi verilmedi

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- A boy and a woman in Arab costume making peace signs .
- A boy and a young woman in a headscarf make peace signs .
- A girl in a headdress and a boy with a bandanna are giving the peace sign to the camera .
- A woman wearing a blue headcover and a boy wearing a scarf hold fingers in peace signs .
- A woman with a blue headdress and a boy with a headband make a peace sign with their fingers

## 17 — 229954612_a4438297ee.jpg

**Görüntü kimliği:** `229954612_a4438297ee.jpg`

**Beam-3:** a group of people are walking through a river

**Etiketler:** eylem

**Gerekçe:** İnsanlar suyun içinden değil köprü üzerinden geçiyor; through a river mekânsal ilişkiyi yanlış veriyor.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- People gather on a bridge near a weeping willow tree .
- People on a bridge look at the water .
- People stand on a bridge over water .
- People walk on a small bridge over a lake .
- Some people are crossing a green bridge over a river .

## 18 — 3265864834_e0229020dd.jpg

**Görüntü kimliği:** `3265864834_e0229020dd.jpg`

**Beam-3:** a brown dog is running on the beach

**Etiketler:** belirsiz — eylem

**Gerekçe:** Görüntü baş sallama veya silkelenmeyle uyumlu. Bir referans koşmayı tarif ediyor ve tek kareden eylem kesinleştirilemiyor

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Not / belirsizlik:** Görüntü baş sallama veya silkelenmeyle uyumlu. Bir referans koşmayı tarif ediyor ve tek kareden eylem kesinleştirilemiyor

**Beş referans:**

- A brown dog making a funny face while standing on a beach .
- A brown dog shaking its head .
- A brown dog wearing a chain around its neck is shaking it head .
- a brown dog wearing a chain collar on a mountain running crazily .
- A dark brown dog rapidly shaking off .

## 19 — 197504190_fd1fc3d4b7.jpg

**Görüntü kimliği:** `197504190_fd1fc3d4b7.jpg`

**Beam-3:** a boy in a green shirt is running through a field

**Etiketler:** renk, genel ifade

**Gerekçe:** Üst giysiler açık gri ve kırmızı görünüyor. Tahmin ikinci çocuğu, topu ve futbol bağlamını atlıyor. Koşma, futbol sırasında mümkün olduğu için kesin eylem hatası eklenmedi

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- Two boys in a field kicking a soccer ball .
- Two children are playing with a soccer ball on grass .
- Two children playing with a ball on the grass .
- Two children play soccer in the park .
- Two little boys are playing outside with their soccer ball on the green grass .

## 20 — 3415578043_03d33e6efd.jpg

**Görüntü kimliği:** `3415578043_03d33e6efd.jpg`

**Beam-3:** a little boy in a yellow shirt is swimming in water

**Etiketler:** yanlış nesne, eylem

**Gerekçe:** Tahmin, boya kaplı zemini su olarak betimliyor ve oturan çocuğun yüzdüğünü söylüyor. Giysi boya ile kaplı olduğundan özgün rengi belirlenemedi

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- A child covered with paint sits amid trays of multicolored paint on a surface thoroughly saturated with paint .
- A child plays with paint .
- A little girl in a white shirt has been playing with paint and covered herself and her room .
- A young child sitting in a mess of paint .
- a young child sitting in a mess of paint spilled everywhere .

## 21 — 1119463452_69d4eecd08.jpg

**Görüntü kimliği:** `1119463452_69d4eecd08.jpg`

**Beam-3:** a girl in a swimming pool

**Etiketler:** Bu altı kategoride kesin etiket yok

**Gerekçe:** Havuzdaki çocuk betimlemesi görüntü ve referanslarla uyumlu; kısa olması tek başına genel ifade hatası sayılmadı.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A girl in a pink swimsuit in a pool
- A girl swimming in a pool .
- A little girl in a pink bathing suit swimming in a pool .
- A little girl swimming in a pool .
- Smiling little girl swimming in outdoor pool .

## 22 — 2594042571_2e4666507e.jpg

**Görüntü kimliği:** `2594042571_2e4666507e.jpg`

**Beam-3:** a young girl wearing a hat and a white shirt is holding a baby

**Etiketler:** yanlış nesne, renk, eylem

**Gerekçe:** Öndeki çocuk turuncu üstlü; white shirt yanlış. Görüntüde kucakta baby yok, holding a baby eylemi de yok; arkada başka çocuk var.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Not / belirsizlik:** Cinsiyet tahmini etiketlenmedi; sayı ayrıca verilmedi çünkü tahmin zaten iki kişi ima ediyor.

**Beş referans:**

- A boy with an orange shirt smiles , while a boy in a blue shirt looks on .
- A little boy in a tan hat on the ground of a play area .
- A smiling boy in an orange shirt with a safari hat sits at the playground , while a boy in a blue shirt looks on .
- Boy in tan hat laughs while sitting near playground equipment .
- Two boys , one wearing a hat , in a playground .

## 23 — 3220126881_b0a4f7cccb.jpg

**Görüntü kimliği:** `3220126881_b0a4f7cccb.jpg`

**Beam-3:** a man in a black shirt and a black leather jacket is sitting on a bench

**Etiketler:** yanlış nesne, genel ifade, belirsiz — eylem

**Gerekçe:** Tahmin diğer iki kişiyi ve birlikte poz verme bağlamını atlıyor, kadrajda görünmeyen bir bank ekliyor. Referanslar ayakta durmayı destekliyor ancak kırpım nedeniyle duruş görselden kesinleştirilemiyor

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Not / belirsizlik:** Tahmin diğer iki kişiyi ve birlikte poz verme bağlamını atlıyor, kadrajda görünmeyen bir bank ekliyor. Referanslar ayakta durmayı destekliyor ancak kırpım nedeniyle duruş görselden kesinleştirilemiyor

**Beş referans:**

- Three people stand by a doorway posing for a photograph .
- Two men and a woman pose for a picture .
- Two men and a woman smile at the camera .
- Two men and a woman standing near a doorway .
- Two men and a woman with their arms around each other .

## 24 — 3163563871_cef3cf33ea.jpg

**Görüntü kimliği:** `3163563871_cef3cf33ea.jpg`

**Beam-3:** a tennis player is playing tennis

**Etiketler:** Bu altı kategoride kesin etiket yok

**Gerekçe:** Tahmin kısa olsa da görüntünün ana etkinliğini doğru anlatıyor. Vuruş ve kıyafet ayrıntılarının atlanması hata sayılmadı

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- A boy in a white shirt and blue shirts is playing tennis .
- A boy swings his tennis racket at a tennis ball .
- A kid in a black cap and light blue shorts playing tennis .
- A person with sunglasses , blue shorts and a white shirt hitting a tennis ball .
- A woman is hitting a tennis ball .

## 25 — 2600883097_aca38cc146.jpg

**Görüntü kimliği:** `2600883097_aca38cc146.jpg`

**Beam-3:** two brown dogs play in the water

**Etiketler:** Bu altı kategoride kesin etiket yok

**Gerekçe:** İki kahverengi köpek sığ suda oynuyor; sayı, renk ve eylem uyumlu.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- two brown dogs playing some water
- Two brown dogs play in shallow water
- Two brown dogs play in the water .
- Two dogs ankle-deep in water .
- Two dogs in shallow water .

## 26 — 3084711346_fda0f5a3e6.jpg

**Görüntü kimliği:** `3084711346_fda0f5a3e6.jpg`

**Beam-3:** a man wearing a helmet

**Etiketler:** genel ifade

**Gerekçe:** Kask doğru; elindeki megafon ve birden çok kişinin belirgin bağlamı atlanmış. Tek kişiyi betimlemek tek başına sayı hatası sayılmadı.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A man in a red helmet holds a bullhorn .
- A man next to a woman holding a megaphone .
- A person holds a blow horn while wearing a helmet .
- Two men are wearing helmets , while one of them holds a bullhorn .
- Two people wearing handkerchiefs over their faces , one with a blow horn

## 27 — 489065557_0eb08889cd.jpg

**Görüntü kimliği:** `489065557_0eb08889cd.jpg`

**Beam-3:** a man in a green shirt is jumping into a lake

**Etiketler:** yanlış nesne

**Gerekçe:** Suya atlayan kişinin üst gövdesi çıplak; green shirt uydurulmuş. Atlayış doğru, olmayan giysiye ayrıca renk etiketi verilmedi.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A boy jumping from a rope into the water .
- A boy wearing red swim trunks is swinging from a rope into the water .
- A man has let go of the rope swing over the river .
- A person in swimming trunks who has used a rope swing to jump into a lake .
- The man is dropping from a swinging rope into a calm river .

## 28 — 3359563671_35b67898e7.jpg

**Görüntü kimliği:** `3359563671_35b67898e7.jpg`

**Beam-3:** a man is doing a trick

**Etiketler:** genel ifade

**Gerekçe:** Havada snowboard hareketi var; doing a trick ekipman ve kar sporunu belirtmiyor.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A person jumping on their snowboard .
- A snowboarder in black and yellow is jumping on a purple snowboard .
- A young man is doing a ski jump with a purple board .
- Snowboarder in yellow pants and white helmet in midair .
- Snowboarder holding onto board while in midair .

## 29 — 3553225222_f5ebe44af1.jpg

**Görüntü kimliği:** `3553225222_f5ebe44af1.jpg`

**Beam-3:** a man in a blue shirt and a blue shirt

**Etiketler:** yanlış nesne, renk, genel ifade, kelime tekrarı

**Gerekçe:** Çocuk kırmızı mezuniyet cübbesiyle kep atıyor; blue shirt iki kez yineleniyor ve mezuniyet/eylem yok.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A child in a graduation gown throws the cap in the air .
- A child who is celebrating their graduation in a field .
- A young child dressed in a graduation gown is throwing a graduation cap in the air .
- A young girl throws her graduation cap into the air .
- Young girl in red graduation suit flings her hat upward into the field .

## 30 — 3270273940_61ef506f05.jpg

**Görüntü kimliği:** `3270273940_61ef506f05.jpg`

**Beam-3:** a boy in a blue shirt is climbing a rope

**Etiketler:** yanlış nesne, renk, eylem

**Gerekçe:** Çocuklar basamak kenarından aşağı atlıyor. Tahmin ip tırmanışı anlatıyor ancak tırmanılan bir ip görünmüyor. Üst giysiler kırmızı ve pembe tonlarında, mavi parçalar ise pantolonlar

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- Children jumping off of cement .
- Two children are jumping off a wall .
- Two children standing on a wall .
- Two young children jumping off of a cement ledge onto the dirt ground .
- Two young girls are jumping off of a steep step .

## 31 — 3488087117_2719647989.jpg

**Görüntü kimliği:** `3488087117_2719647989.jpg`

**Beam-3:** a white and white dog is playing with a toy

**Etiketler:** yanlış nesne, eylem, kelime tekrarı

**Gerekçe:** Köpek kaldırımda koşuyor; görünür oyuncak yok. white and white yinelenmiş. Kahverengi ayrıntıların atlanmasına renk etiketi verilmedi.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- a brown and white dog jumps on the sidewalk .
- A little white dog in running on the sidewalk .
- A white dog with brown ears is running on the sidewalk .
- Little brown and white dog running on the sidewalk .
- The dog is running .

## 32 — 2933637854_984614e18b.jpg

**Görüntü kimliği:** `2933637854_984614e18b.jpg`

**Beam-3:** a football player in a blue uniform

**Etiketler:** renk, genel ifade

**Gerekçe:** Forma turuncu, mavi değil; eldeki top ve hazırlık eylemi atlanmış.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- a football player wearing an orange jersey holding a football
- A football player wearing an orange uniform smiling and holding the football .
- A man in an orange jersey with the letter " 12 " on it plays football .
- A University of Miami player holds a football .
- Number 12 in orange gets ready to throw the football .

## 33 — 2854959952_3991a385ab.jpg

**Görüntü kimliği:** `2854959952_3991a385ab.jpg`

**Beam-3:** a white and white dog is running through the grass

**Etiketler:** eylem, genel ifade, kelime tekrarı

**Gerekçe:** Tahmin ikinci köpeği ve top yakalama bağlamını atlıyor. Köpeğin topa doğru sıçramasını koşma olarak betimliyor. ‘White and white’ ifadesinde aynı renk gereksiz yere yineleniyor

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- One dog leaps to catch a softball while another looks on .
- The two dogs , one of which is leaping in the air , are chasing a ball outdoors .
- two dogs jump for a ball on a green field .
- Two dogs play catch with a ball in a field .
- Two small dogs pursuing a ball across a grassy field .

## 34 — 3097171315_0ba7d283b1.jpg

**Görüntü kimliği:** `3097171315_0ba7d283b1.jpg`

**Beam-3:** a group of people sit on a bed

**Etiketler:** yanlış nesne, eylem

**Gerekçe:** Grup gösterişli odada ayakta; ön planda kanepe var, bed yok. Grup sit eyleminde değil.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- a group of people are congregated in a room by a big round mirror .
- A group of people enter a fancy golden living room .
- A group of people mills about in a room with crystal chandelier , a brown leather couch and a large round mirror .
- A group of people stand in a fancily decorated room .
- The people are gathered in a room with a sofa , round mirror , and chandelier .

## 35 — 3085667865_fa001816be.jpg

**Görüntü kimliği:** `3085667865_fa001816be.jpg`

**Beam-3:** a man is surfing on a surfboard

**Etiketler:** Bu altı kategoride kesin etiket yok

**Gerekçe:** Tahmin ana etkinlik olan sörfü doğru anlatıyor. Havadaki hareket belirtilmediği için açıklama daha az ayrıntılı, ancak bu eksiklik kesin hata olarak sayılmadı

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- a lone surfboarder jumping a wave on a white surfboard .
- A man flies into the air with his surfboard .
- A person in a wetsuit is in midair while surfing .
- A surfer is in the air above the ocean .
- A surfer performs a trick .

## 36 — 3530843182_35af2c821c.jpg

**Görüntü kimliği:** `3530843182_35af2c821c.jpg`

**Beam-3:** a man wearing a blue shirt and a blue shirt

**Etiketler:** genel ifade, kelime tekrarı

**Gerekçe:** Tahmin ikinci kişiyi ve kırmızı arabayı atlıyor. ‘A blue shirt’ öbeği gereksiz yere yineleniyor. Gömlekte mavi çizgiler görüldüğü için kesin renk hatası eklenmedi

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Beş referans:**

- A couple practice different ways to wear sunglasses in front of a red car .
- A girl and boy with sunglasses and a red car behind them .
- A man with sunglasses and a woman with a hat are in front of a car .
- a man with sunglasses on his head is looking at the woman with a white hat and red scarf next to him whilst standing in front of a red sports car .
- A woman in a crochet hat and sunglasses smiles while the man beside her in a plaid short and sunglasses looks behind her .

## 37 — 418796494_bdb441de42.jpg

**Görüntü kimliği:** `418796494_bdb441de42.jpg`

**Beam-3:** a black and white dog with a red collar

**Etiketler:** yanlış nesne, renk, genel ifade

**Gerekçe:** Köpek siyah/kahverengi, black and white değil; red collar görünmüyor ve yatak bağlamı atlanmış.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- A dachshund puppy jumps on a bed .
- A puppy jumps on a bed with purple sheets .
- A puppy on a bed with blue sheets and pillow .
- A small black and brown puppy is looking up
- The dog is on the bed .

## 38 — 3052038928_9f53aa2084.jpg

**Görüntü kimliği:** `3052038928_9f53aa2084.jpg`

**Beam-3:** a group of people sit on a bench

**Etiketler:** belirsiz — yanlış nesne

**Gerekçe:** İki kişi gerçekten oturuyor. Görsel tarif ve referanslar kanepeyi destekliyor, ancak mobilyanın türü kesinleştirilemedi

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Not / belirsizlik:** İki kişi gerçekten oturuyor. Görsel tarif ve referanslar kanepeyi destekliyor, ancak mobilyanın türü kesinleştirilemedi

**Beş referans:**

- A group of five young adults lounging indoors .
- A man and a woman sit on a couch and look at each other while three other men stand by .
- A small group of people standing while two sit on the couch .
- There are four men and one woman .
- Three men are setting up a scene by measuring the lights for a couple sitting on a low couch .

## 39 — 374828031_9d087da5cf.jpg

**Görüntü kimliği:** `374828031_9d087da5cf.jpg`

**Beam-3:** a man climbs a rock wall

**Etiketler:** Bu altı kategoride kesin etiket yok

**Gerekçe:** Kişi kaya duvarına tırmanıyor; ana nesne ve eylem doğru.

**Kullanıcı kontrolü:** İlk görsel incelemede sonuçlandırıldı.

**Beş referans:**

- a climber is scaling a reddish colored rock face .
- a lone man climbing a huge rock wall
- A man climbing a cliff .
- A man climbing a rock cliff .
- The man is scaling a large copper boulder .

## 40 — 2599903773_0f724d8f63.jpg

**Görüntü kimliği:** `2599903773_0f724d8f63.jpg`

**Beam-3:** a little girl blowing bubbles

**Etiketler:** belirsiz — kişi niteliği (boy/girl)

**Gerekçe:** Baloncuk üfleme eylemi doğru. Tahmindeki ‘girl’ ifadesi beş referanstaki ‘boy’ ifadesiyle uyuşmuyor. Kişi niteliği görüntüden doğrulanamadı

**Kullanıcı kontrolü:** Tamamlandı; karar kaydedildi.

**Not / belirsizlik:** Baloncuk üfleme eylemi doğru. Tahmindeki ‘girl’ ifadesi beş referanstaki ‘boy’ ifadesiyle uyuşmuyor. Kişi niteliği görüntüden doğrulanamadı

**Beş referans:**

- A boy in a plaid shirt blowing bubbles with a yellow bubble wand .
- A boy in over-alls blowing bubbles in a field .
- A boy in overalls blows bubbles .
- A young boy blows bubbles on a grassy lawn .
- Little boy in plaid shirt blows bubbles outdoors .