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

* Örnekler tamamiyle, modelin yanlış tahmin yaptığı örneklerden oluşur. Bu örnekler, modelin neden hatalı tahmin yaptığını anlamak için incelenir.
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

### Örneklemler: 

* Örneğin: "Berlin düşüp faşizm kaybettikten sonra Kızıl Ordu askerlerinin teca.. pardon özgürleştirdiği kadınlar..." ifadesi, model tarafından RACIST olarak etiketlenmiştir. Ancak bu ifade, ırkçılıkla ilgili aşağılayıcı bir söylem içermemektedir.

https://user-images.githubusercontent.com/83168207/228993692-bf1e10e9-4371-4878-a5a1-f705113be056.mov

Model tarafından "Berlin düşüp faşizm kaybettikten sonra Kızıl Ordu askerlerinin teca.. pardon özgürleştirdiği kadınlar..." ifadesinin RACIST olarak etiketlenmesi şaşırtıcı bir durumdur. Shap analizi sonuçlarına bakıldığında, örneklemin yeteri kadar net olmadığı ve eksik bir tanımdan dolayı böyle bir etiketleme yapıldığı görülmektedir.

Shap analizi, modelin karar verme sürecindeki her öznitelik için bir önem derecesi belirler. Bu örnekte, "ordu" için 0.192, "özgür" için 0.131 ve "kaybettikten" için 0.324 önem değerleri elde edilmiştir. Base value ise modelin, bir veri noktasının sınıflandırma etiketini "RACIST" olarak belirleme olasılığının 0.000951051 olduğunu göstermektedir.

Sonuç olarak, Shap analiz sonuçları, modelin "Berlin düşüp faşizm kaybettikten sonra Kızıl Ordu askerlerinin teca.. pardon özgürleştirdiği kadınlar..." ifadesini yanlış şekilde "RACIST" olarak etiketlediğini göstermektedir. Analiz sonuçları, modelin sınıflandırma kararlarını daha doğru hale getirmek için daha net ve kapsamlı eğitim verileri kullanılması gerektiğini vurgulamaktadır.

---
