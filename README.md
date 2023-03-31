<img width="1595" alt="Ekran Resmi 2023-03-31 02 31 14" src="https://user-images.githubusercontent.com/83168207/228987213-58fcc670-f474-46e1-b474-6a4c74a316e1.png">

--- 
## Shap Analizi ve Renklerin Anlamı

Teknofest 2023 Doğal Dil İşleme yarışması için gerçekleştirilen bu çalışma, Shap Analizi yöntemi kullanılarak modelin tahminlerinin nasıl oluşturulduğunu açıklamaktadır. Bu çalışmanın amacı, modelin tahminlerine daha fazla şeffaflık kazandırmaktır.

Shap analizinde **mavi** ve **kırmızı** renkler, her bir özellik veya kelimenin pozitif veya negatif etkisini göstermektedir. 

- **Mavi renk**, bir özellik veya kelimenin sınıfın tahmin edilmesine katkıda bulunmak için **olumsuz** bir etkiye sahip olduğunu gösterirken, **kırmızı renk olumlu** bir etkiye sahip olduğunu gösterir.

**Örneğin,** bir cümlede "harika" kelimesi **kırmızı** renkte gösterilirse, bu kelimenin tahmin edilen sınıf için olumlu bir etkiye sahip olduğu anlamına gelir. Benzer şekilde, bir cümlede "kötü" kelimesi **mavi** renkte gösterilirse, bu kelimenin tahmin edilen sınıf için olumsuz bir etkiye sahip olduğu anlamına gelir.

Bu nedenle, **mavi** renkli özellikler ve kelimeler, tahmin edilen sınıfı düşürmek için katkıda bulunan önemli özellikleri ve kelimeleri gösterirken, **kırmızı** renkli özellikler ve kelimeler, tahmin edilen sınıfı artıran önemli özellikleri ve kelimeleri gösterir.

---

### Aşağıdaki Tabloda Yer Alan Örnekler ile Modelimizin Sınıflandırma Performansını SHAP Analizi Yöntemiyle Değerlendiriyoruz:

* Örneklerin bir kısmı modelin yanlış tahmin yaptığı örneklerden oluşur. Bu örnekler, modelin neden hatalı tahmin yaptığını anlamak için incelenir.
* Tercih edilen örneklerin tamamına ulaşmak için lütfen [tıklayınız.](https://github.com/Teknofest-Nane-Limon/nlp-shap-exploration-teknofest-2023/blob/main/dataset/shap.csv)

|text|target|
|----|------|
|Sakallı lavuk trans kız olduğunu iddia eden tweet atar Maksimum 30takipcili Sünni hetero anon erkekler:|RACIST|
|48 kilo kizmi olur amk kadın dediğin ele avuca gelicek haksizmiyim|PROFANITY|
|Ay bokum 1.60 altı kadınlar kendine kadınım demesin|PROFANITY|
|Bakire kız arıyorsan sen de bakir erkek olacaksın. Milli olmak uçkuru bozuk güruhun uydurması.. her kültür doğru değil|PROFANITY|
|Berlin düşüp faşizm kaybettikten sonra Kızıl Ordu askerlerinin teca.. pardon özgürleştirdiği kadınlar...|RACIST|
|biz bu hayatta hic esmer 1.80+ yakisikli cocuklardan cekmedik. hep kumral 1.70 yer cucesi buyuyememis sivilceli suratli ergenlerden cektik|OTHER|

---


### Model Performansının Analizi ve İyileştirilmesi için SHAP Değerleri Hesaplanarak Görselleştiriliyor

- Öncelikle, `sp.Explainer()` fonksiyonu ile SHAP değerleri hesaplayacağız ve ardından `sp.plots.text()` fonksiyonu ile görselleştireceğiz.

- `sp.plots.text()` fonksiyonunun çıktısı, her bir özniteliğin önem derecesini gösteren bir görsel olarak sunulur. Bu görsel, metin sınıflandırma modellerinde özellikle faydalıdır çünkü modelin hangi özelliklere (kelimelere) daha fazla dikkat ettiğini ve tahmin sonuçlarına ne kadar katkıda bulunduklarını gösterir. Bu nedenle, `sp.plots.text()` fonksiyonu, modelin performansını analiz etmek ve iyileştirmek için değerli bir işlevdir.

```python
import shap as sp 

explainer = sp.Explainer(pred)
shap_values = explainer(shap_df['text'][:])
sp.plots.text(shap_values)
```

### Yanlış Etiketlenmiş Örneklemler | RACIST Sınıfı : 

Örneğin: "Berlin düşüp faşizm kaybettikten sonra Kızıl Ordu askerlerinin teca.. pardon özgürleştirdiği kadınlar..." ifadesi, model tarafından RACIST olarak etiketlenmiştir. Ancak bu ifade, ırkçılıkla ilgili aşağılayıcı bir söylem içermemektedir.

https://user-images.githubusercontent.com/83168207/228993692-bf1e10e9-4371-4878-a5a1-f705113be056.mov

Model tarafından "Berlin düşüp faşizm kaybettikten sonra Kızıl Ordu askerlerinin teca.. pardon özgürleştirdiği kadınlar..." ifadesinin RACIST olarak etiketlenmesi şaşırtıcı bir durumdur. Shap analizi sonuçlarına bakıldığında, örneklemin yeteri kadar net olmadığı ve eksik bir tanımdan dolayı böyle bir etiketleme yapıldığı görülmektedir.

Shap analizi, modelin karar verme sürecindeki her öznitelik için bir önem derecesi belirler. Bu örnekte, "ordu" için 0.192, "özgür" için 0.131 ve "kaybettikten" için 0.324 önem değerleri elde edilmiştir. Base value ise modelin, bir veri noktasının sınıflandırma etiketini "RACIST" olarak belirleme olasılığının 0.000951051 olduğunu göstermektedir.

Sonuç olarak, Shap analiz sonuçları, modelin "Berlin düşüp faşizm kaybettikten sonra Kızıl Ordu askerlerinin teca.. pardon özgürleştirdiği kadınlar..." ifadesini yanlış şekilde "RACIST" olarak etiketlediğini göstermektedir. Analiz sonuçları, modelin sınıflandırma kararlarını daha doğru hale getirmek için daha net ve kapsamlı eğitim verileri kullanılması gerektiğini vurgulamaktadır.

---

Bir başka örnek ise; "*Sakallı lavuk trans kız olduğunu iddia eden tweet atar* Maksimum 30takipcili Sünni hetero anon erkekler" Bu ifade, model tarafından "RACIST" olarak yanlış etiketlenmiştir. Analiz sonuçlarına bakıldığında, "Sunni" kelimesinin oldukça yüksek bir önem derecesine sahip olduğu (0.971) ve modelin bu kelimeyi yanlış anladığı görülmektedir.

https://user-images.githubusercontent.com/83168207/229002765-397e10ad-b1cc-4f6d-b8e8-af25e11ba674.mov


Ancak, "Sunni" kelimesinin kullanımı bu ifadede ırkçı veya ayrımcı bir söylemi içermemektedir. Bu ifadede, "Sünni hetero anon erkekler" kavramı, belirli bir gruba yönelik bir açıklama olarak kullanılmıştır ve bu ifade içinde herhangi bir ayrımcı, ırkçı veya nefret söylemi bulunmamaktadır.

Bu nedenle, modelin bu ifadeyi "RACIST" olarak etiketlemesi yanlış bir sonuçtur ve yine bu tür hataların önlenmesi için daha net ve kapsamlı eğitim verilerinin kullanılması gerekmektedir. Modelin, kelime kullanımı içindeki bağlamı daha iyi anlamasını sağlayacak şekilde eğitilmesi gerekmektedir.

---


### Shap Analizi ile RACIST Sınıfındaki Özniteliklerin Etkisi ve Modelin Yanlış Etiketleme Sorunu


Shapley değerleri, makine öğrenmesi modellerindeki her bir özniteliğin sınıflandırma sonucundaki önemini belirlemek için kullanılan bir yöntemdir. Bu değerler, her öznitelik için hesaplanan Shap değerlerinin ortalamasını temsil eder. `Shap.plots.bar()` fonksiyonu ise, bu Shapley değerlerini bar grafiği şeklinde görselleştirir. Bu fonksiyon, her öznitelik için Shap değerlerini bir çubuk grafikte gösterir. Bu grafik, özniteliklerin sınıflandırma kararında ne kadar önemli olduğunu gösterir.

Bu örnekte, RACIST sınıfındaki özniteliklerin etkisini belirlemek için `Shap.plots.bar()` işleviyle incelenmiştir. Analiz sonucunda, "Sunni" kelimesinin en etkili öznitelik olduğu (0.97), diğer öznitelikler arasında "fas kelimesi" (0.49), "abd kelimesi" (0.46), "turk kelimesi" (0.4) ve "araplar kelimesi" (0.44) gibi değerler yakaladığı gözlemlenmiştir:

```python
$   sp.plots.bar(shap_values[:,:,3].mean(0), order= sp.Explanation.argsort.flip);
```

<img width="1026" alt="Ekran Resmi 2023-03-31 05 01 45" src="https://user-images.githubusercontent.com/83168207/229005215-2aaf828c-39b3-4722-84f4-93a848462a83.png">



Bu sonuçlar, modelin sınıflandırma kararını belirlemede "Sunni" kelimesinin diğer özniteliklere göre daha önemli olduğunu gösterir. Ancak, bu sonuçlar da göstermektedir ki, modelin RACIST sınıfını belirlerken sadece bir kelimeye dayanarak yanlış etiketleme yapabileceği.

Özet olarak, daha doğru sınıflandırma kararları almak için modelin daha kapsamlı bir eğitim verisiyle eğitilmesi gerekmektedir. Ayrıca, modelin kelime kullanımı içindeki bağlamı daha iyi anlaması için eğitilmesi de önemlidir.


---

### Sonuç: 

Bu çalışma, Teknofest 2023 Doğal Dil İşleme yarışması için gerçekleştirilen bir SHAP Analizi çalışmasını ele almaktadır. Shap Analizi yöntemi kullanılarak modelin tahminlerinin nasıl oluşturulduğu açıklanmıştır.

Bu çalışma kapsamında yer alan örneklemlerin sadece bir kısmı modelin yanlış tahmin yaptığı örneklerden oluşur. Örneklemler üzerinde yapılan analizin tamamına ulaşmak için notebook'u inceleyebilirsiniz. 

Bunun yanı sıra, çalışmanın sınırlılıkları da göz önünde bulundurulmalıdır. Örneklem verilerinin sınırlı olması nedeniyle, elde edilen sonuçların genelleştirilebilirliği sınırlıdır. Ayrıca, çalışmada kullanılan yöntemin avantajları ve dezavantajları da ayrıca dikkate alınmalıdır. Belirlenen örnekler, incelenerek modelin hatalı tahminlerinin anlaşılmaya çalışıldığı bu çalışma kapsamında, model performansının iyileştirilmesi için SHAP değerleri hesaplanarak görselleştirilmiştir. Bu sayede, RACIST sınıfındaki özniteliklerin etkisi ve modelin yanlış etiketleme sorunu incelenmiştir. Bağımlı değişkenin sınıflarının her biri ayrı ayrı ele alınmış ve incelenmiştir. Shapley değerleri kullanılarak her bir öznitelik için sınıflandırmada ne kadar önemli olduğu belirlenmiştir. Analiz sonuçları, modelin sınıflandırma kararını belirlemede bazı kelime özniteliklerinin diğerlerine göre daha önemli olduğunu gösterirken, modelin sınıflandırma kararlarında hatalı sonuçlar verebildiği gözlemlenmiştir. Gerçekleştirilen SHAP Analizi sonucunda, modelin sınıflandırma kararlarını daha doğru hale getirmek için verisetinin güçlendirilebileceği ve modelin yeniden eğitimlesi gerektiği kanatine ulaşılmıştır. 

Sonuç olarak, Shap Analizi yöntemi kullanılarak modelin sınıflandırma performansının analiz edilmesini ve iyileştirilmesini sağlayan bir çalışma yürütülmüştür... 


SHAP Analizi ile ilgili daha fazla bilgi edinmek için daha önce bu konuyla ilgili yazmış olduğum 
[Medium yazısına](https://medium.com/@tarikkaan1koc/shap-analizi-shap-de%C4%9Ferleriyle-makine-%C3%B6%C4%9Frenimi-modelleri-nas%C4%B1l-yorumlan%C4%B1r-e95710e4aa0c) ve/veya [Shap Library dokümantasyona](https://shap.readthedocs.io/en/latest/index.html) göz atabilirsiniz.


