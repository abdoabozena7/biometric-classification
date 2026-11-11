# Oral Defense: Simple English and Egyptian Arabic

## The One-Minute Explanation

**English:** We have three virtual users. Each has supplied face pictures, one
public speaker, and one public finger. They are not physically the same people.
We turn each input into traditional numerical features, standardize them and
reduce dimensions with PCA fitted only on training data. We compare KNN with
three neighbors and linear SVM for six tasks, giving twelve classifiers. A claim
passes only if the predicted identity matches and its distance is acceptable.
The hierarchy stops at the first rejection. After all five gates pass, the same
voice vector predicts an isolated word. A permission dictionary decides whether
that verified user's role may execute the simulated command.

**بالمصري:** عندنا تلات مستخدمين افتراضيين. لكل واحد صور وش من الداتا اللي
اتقدمت، وصوت متكلم واحد من داتا عامة، وبصمات مختلفة لنفس الصباع من داتا عامة.
بس الوش والصوت والبصمة مش لنفس الشخص الحقيقي. بنحوّل كل عينة لأرقام بطرق
تقليدية، وبنعمل StandardScaler وPCA باستخدام التدريب بس. بنقارن KNN بتلات
جيران مع SVM خطي في ست مهام، يعني اتناشر مصنف. مش كفاية الموديل يقول اسم؛
لازم الاسم يطابق الادعاء والمسافة تبقى جوه الحد. أول مرحلة ترفض بتوقف كل اللي
بعدها. لو التحقق كله نجح، بنستخدم نفس خصائص الصوت عشان نعرف الكلمة، وبعدها
قاموس الصلاحيات يقرر ينفذ أمر محاكاة ولا يرفض.

## Which File Do I Explain First?

Open the main notebook and follow its numbered sections. The functions displayed
in its outputs come directly from the shared modules, so they are the actual
implementation, not a second simplified algorithm. The source files are ordinary
functions and dictionaries, with no custom model framework or unnecessary classes.

بالمصري: افتح النوتبوك الأساسي وامشي بالترتيب. الكود المعروض للخصائص والتحقق
هو نفس الكود اللي التطبيق بيستخدمه، مش مثال مختلف. ملفات بايثون موجودة عشان
النوتبوك والتطبيق يشتركوا في نفس الدوال.

## Every Module: Input, Work, Output, Caller, Next Step, Technique, Why

### config.py

**English:** Input: the project location. Work: define paths, random seed,
supported words and role IDs. Output: constants and created folders. Called by:
all modules. Next: each module uses the same paths and labels. Technique: no ML.
Why: one obvious place for configuration prevents conflicting settings.

**بالمصري:** المدخل هو مكان المشروع. الملف بيحدد المسارات والـseed والكلمات
والأدوار، وبيطلع إعدادات مشتركة. كل الملفات بتناديه وبتستخدم نفس القيم بعده.
مفيهوش تعلم آلي؛ وجود الإعدادات في حتة واحدة بيمنع اللخبطة.

### dataset_setup.py

**English:** Inputs: three face folders and real public source files. Work:
decode, hash, reject exact duplicates/corruption, warn on similar faces, select
consistent speakers/fingers, split and reserve demo files. Outputs: samples.csv,
identities.json, source/count audits, pair/triplet manifests and live_test.
Called by: the notebook or command line. Next: train.py consumes the manifests.
Technique: deterministic shuffle and data validation, not a biometric classifier.
Why: prevent leakage, fabricated data and inconsistent virtual identities.

**بالمصري:** المدخل صور التلات أشخاص والداتا العامة الحقيقية. بيفتح الملفات،
يحسب الهاش، يشيل التالف والتكرار الكامل، وينبه للصور المتشابهة. بيختار متكلم
واحد وصباع واحد لكل مستخدم ويقسم الداتا ويحجز عينات العرض. الناتج ملفات
manifests وlive_test. النوتبوك أو الطرفية بينادوه، وبعده train.py بياخد
المانيفست. دي تجهيزات داتا وتقسيم ثابت، مش موديل تعرف على الأشخاص.

### features.py

**English:** Input: an image/audio path or bytes. Work: preprocess consistently
and extract HOG/LBP/color or MFCC/ZCR/RMS. Output: finite numerical vectors, or
an error for unsupported input. Called by: train.py and verification.py. Next:
the training-fitted scaler/PCA receives the vector. Techniques: traditional
handcrafted features. Why: simple, interpretable and compliant with the assignment.

**بالمصري:** بياخد مسار ملف أو bytes، ويجهز الصورة أو الصوت بنفس الطريقة
كل مرة. بيطلع متجه أرقام، أو خطأ لو العينة مش صالحة. التدريب والتحقق بينادوه،
والأرقام بتروح للـScaler والـPCA. الطرق يدوية وتقليدية عشان نفهمها بسهولة
ومنستخدمش تعلم عميق.

### train.py

**English:** Inputs: split manifests and extracted features. Work: fit each
scaler/PCA only on training, train KNN and linear SVM, compare validation macro-F1,
then accuracy, and calibrate validation distance thresholds. Outputs: twelve
fitted classifiers, preprocessors, centroids, thresholds, selected-model index
and validation comparison. Called by: the notebook or CLI. Next: verification,
evaluation and app load the saved artifacts. Why: two basic model families make
an understandable comparison; test data cannot influence selection.

**بالمصري:** بياخد الداتا المقسومة وخصائصها، ويدرّب الـScaler والـPCA على
التدريب بس. بعد كده يدرّب KNN وSVM، ويختار من نتيجة الـvalidation، ويحدد حدود
المسافة من نفس الـvalidation. بيحفظ اتناشر مصنف والتحويلات والمتوسطات والحدود.
النوتبوك أو الطرفية بينادوه، وبعده التحقق والتقييم والتطبيق يحملوا الملفات.
الاختيار مش على الاختبار عشان منغشش النتيجة.

### fusion.py

**English:** Inputs: transformed modality vectors and explicit same-user,
same-split pairs/triplets. Work: concatenate actual feature values. Output: one
longer numerical vector/matrix. Called by: training, verification and hierarchy.
Next: the dedicated face_voice or fusion3 classifier. Technique: early feature
fusion, not voting between predicted names. Why: the model learns from combined
measurements, while each source impression is counted honestly.

**بالمصري:** بياخد متجهات الخصائص بعد التحويل لنفس المستخدم ونفس الجزء من
الداتا. بيلزق الأرقام جنب بعض وبيطلع متجه أطول، وبعده مصنف fusion متدرّب
لوحده. التدريب والتحقق والبايبلاين بينادوه. إحنا مش بنجمع أسماء الموديلات؛
بنجمع الخصائص نفسها عشان الموديل يتعلم من المعلومات مع بعض.

### verification.py

**English:** Inputs: sample/vector, claimed identity and saved model. Work:
load trusted local artifacts; transform identically; predict identity; compute
distance to the claimed user's training centroid; compare the fixed threshold.
Output: accepted, claim, prediction, distance, threshold, explanation. Called by:
the hierarchy, notebook and evaluation. Next: continue only when accepted.
Techniques: selected KNN/SVM plus Euclidean distance. Why: classification alone
always names an enrolled class and cannot establish a particular claim.

**بالمصري:** بياخد العينة والهوية اللي الشخص بيدعيها والموديل المحفوظ. بيطبق
نفس التحويل، يتوقع الهوية، ويحسب المسافة لمتوسط تدريب صاحب الادعاء. لازم
التوقع يطابق والمسافة متعديش الحد. بيطلع قاموس فيه القرار وسببه والتفاصيل.
البايبلاين والنوتبوك والتقييم بينادوه؛ لو قبل نكمل، ولو رفض نقف. المصنف لوحده
هيختار اسم حتى لو العينة مش شبه حد، عشان كده ضفنا اختبار المسافة البسيط.

### commands.py

**English:** Input: the SAME transformed voice vector already used for speaker
verification. Work: classify ON/OFF/GO/STOP/UNKNOWN and reject a distant predicted
class. Output: recognized command, supported flag, distance and threshold. Called
by: pipeline.py only after biometric success; evaluation can measure it separately.
Next: permissions.py; only then execute_action may change the simulated state.
Its threshold is calibrated on the exact supported/UNKNOWN validation rule,
separately from claimed-identity thresholds.
Technique: a separate KNN/SVM word classifier using the same MFCC/ZCR/RMS features.
Why: who spoke and what was said are two different labels, not two feature sets.

**بالمصري:** بياخد نفس متجه الصوت اللي استخدمناه للمتكلم، ويصنف الكلمة أو
يرجع UNKNOWN. بيطلع الكلمة وقرار دعمها والمسافة والحد. البايبلاين بينادي
التعرف بعد نجاح التحقق كله، والتقييم يقدر يقيسه منفصل. بعده الصلاحيات، وبعد
السماح بس يحصل تغيير في حالة المحاكاة. معرفة المتكلم ومعرفة الكلمة مهمتين
مختلفتين، حتى لو خصائص الصوت واحدة. ده مش speech-to-text عام.

### permissions.py

**English:** Inputs: verified user ID and recognized word. Work: get that user's
role and check a Python set in a dictionary. Output: ALLOWED/DENIED. Called by:
pipeline.py after successful verification and supported command recognition.
Next: simulated action only if allowed. Technique: no ML. Why: access policy is
explicit, predictable and testable, not something a classifier should guess.

**بالمصري:** بياخد هوية اتحققنا منها والكلمة المتوقعة. بيجيب الدور ويشوف
الكلمة في قاموس الصلاحيات. بيطلع سماح أو رفض. البايبلاين بيناديه بعد التحقق
والتعرف، وبعده التنفيذ لو مسموح. مفيش تعلم آلي هنا؛ الصلاحيات قواعد واضحة.

### pipeline.py

**English:** Inputs: face, voice, fingerprint, claim, models and simulated state.
Work: run face, speaker, face+voice, fingerprint and fusion3 gates in that exact
order; stop on first rejection; reuse audio features for word recognition;
reject UNKNOWN; check role; then simulate the action. Output: complete decision
trace and final state. Called by: notebook, app and evaluation. Next: display or
save the trace. Technique: control flow around trained models, not a seventh
classifier. Why: no protected action may bypass a failed verification gate.

**بالمصري:** المدخل التلات عينات والادعاء والموديلات وحالة المحاكاة. بيشغل
الوش، المتكلم، fusion الاتنين، البصمة، وبعدين fusion التلاتة بالترتيب. أول
رفض بيوقف الباقي. بعد النجاح يستخدم نفس خصائص الصوت للكلمة، يرفض UNKNOWN،
يراجع الصلاحية، وبعدين ينفذ محاكاة. الناتج trace كامل والحالة النهائية.
النوتبوك والتطبيق والتقييم بينادوه ويعرضوا الناتج. ده تنظيم مراحل، مش مصنف جديد.

### evaluate.py

**English:** Inputs: frozen models and actual held-out source files. Work:
re-extract using saved transforms, compare truth/prediction, run genuine/impostor
claims and measure hierarchy/actions. Outputs: CSV/JSON metrics, confusion-matrix
CSV/PNG files and real traces. Called by: notebook or CLI. Next: report and UI
read the measured files. Technique: metrics and inference only; no fitting.
Why: keep test measurement separate from model/threshold selection.

**بالمصري:** بياخد الموديلات الثابتة وملفات الاختبار الحقيقية، ويطلع خصائصها
بالتحويلات المحفوظة من غير تدريب جديد. يقارن الحقيقة بالتوقع، ويجرب الادعاء
الصح والغلط، ويحسب نتائج البايبلاين والأوامر. بيطلع جداول وصور confusion matrix
وtraces. النوتبوك أو الطرفية بينادوه، والتقرير والتطبيق يعرضوا الناتج.
مبنغيرش الموديل أو الحد عشان نتيجة الاختبار تبقى أحسن.

### app.py and the Two Notebooks

**English:** App inputs: chosen source person, claim and offline files or new
camera bytes. Work: preview, call the saved-model hierarchy, display decisions.
Outputs: the actual UI trace and measured tables. Called by: Streamlit server;
the second notebook explains and can launch it. Next: human inspection, not real
equipment. Technique: UI only, no retraining. The main notebook teaches and runs
each pipeline step. Why: keep one implementation while making the workflow visible.

**بالمصري:** التطبيق بياخد الشخص والادعاء والعينات، أو لقطة كاميرا جديدة
للوش بس. يعرضهم وينادي البايبلاين بالموديلات المحفوظة ويعرض القرار. Streamlit
بيشغله والنوتبوك التاني يشرحه ويقدر يفتحه. الناتج للعرض والمحاكاة، مش للتحكم
في جهاز حقيقي. النوتبوك الأساسي بيمشي في كل خطوة، من غير ما نعمل خوارزمية
مختلفة للتطبيق.

### tests/test_system.py

**English:** Inputs: real data/artifacts plus clearly isolated routing doubles.
Work: check permissions, blocking, data splits, source consistency, feature
concatenation and actual UI/model inference. Output: real pytest pass/fail results.
Called by: pytest and the main notebook. Next: confidence in tested contracts,
not an invented biometric score. Technique: assertions, not ML. Why: a perfect
unit-test count cannot establish real-world authentication quality.

**بالمصري:** الاختبارات بتاخد الملفات والموديلات الحقيقية، ومعاهم بدائل
اختبار للتحكم في ترتيب المراحل بس. بتراجع المنع والصلاحيات وتقسيم الداتا
والـfusion وتشغيل التطبيق فعلا. pytest والنوتبوك بينادوها، والناتج نجاح أو
فشل اختبارات. نجاح الاختبارات مش معناه إن دقة الأوامر ممتازة أو النظام آمن للإنتاج.

## The Feature and ML Vocabulary

| Concept | Simple English | بالمصري |
| --- | --- | --- |
| Feature extraction | Turn an input into useful numerical measurements. | نحول الصورة أو الصوت لأرقام بتوصفه. |
| HOG | Count local edge directions; describes shape. | بيعد اتجاهات الحواف في أجزاء صغيرة عشان يوصف الشكل. |
| LBP | Compare a gray pixel with neighbors and count local texture patterns. | يقارن البكسل بجيرانه ويعد أنماط الملمس. |
| Color histogram | Count intensities separately for R, G and B, then normalize. | يعد درجات كل لون من التلاتة ويقسم على العدد عشان الأحجام تبقى قابلة للمقارنة. |
| MFCC | A traditional compact description of speech spectrum using a mel filterbank and cosine transform. | وصف تقليدي لطيف الصوت بطريقة قريبة من توزيع إحساس السمع بالترددات، مش شبكة عصبية. |
| ZCR | Rate at which the signal changes sign around zero. | معدل عبور إشارة الصوت للصفر من موجب لسالب والعكس. |
| RMS | Root mean square amplitude, measuring signal energy. | جذر متوسط مربع شدة الإشارة؛ بيدينا مقياس لطاقة الصوت. |
| StandardScaler | Subtract training means and divide by training standard deviations. | نطرح متوسط التدريب ونقسم على انحرافه المعياري. |
| PCA | Keep the main training-data variation directions using fewer dimensions. | نحتفظ باتجاهات التباين المهمة في التدريب بعدد أبعاد أقل. |
| KNN | Predict the majority class among the three nearest training neighbors. | نشوف أقرب تلات عينات تدريب وناخد تصويتهم. |
| Linear SVM | Learn linear separating boundaries between classes. | يتعلم حدود خطية تفصل الفئات عن بعض. |
| Classification | Assign one of the known target labels. | نختار فئة من الفئات اللي اتعلمناها. |
| Identification | Ask which enrolled person the model predicts. | نسأل الموديل: ده مين من المسجلين؟ |
| Verification | Test a specific identity claim, with rejection possible. | نختبر ادعاء شخص معين ونقدر نرفضه. |
| Early fusion | Concatenate measurements before classifier prediction. | نجمع أرقام الخصائص قبل التوقع، مش أسماء النتائج. |
| Three-modality fusion | Concatenate face, voice and fingerprint feature blocks. | نجمع خصائص الوش والصوت والبصمة في متجه واحد. |
| Hierarchical verification | Ordered gates; any failure stops later processing. | بوابات بالترتيب، أول رفض يوقف اللي بعده. |
| Command recognition | Classify a recorded isolated word, including UNKNOWN. | نعرف كلمة منفصلة من تسجيل، أو نرجع UNKNOWN. |
| Authorization | Check what an already verified role is allowed to do. | نحدد صاحب الهوية اللي اتحققنا منها مسموح له يعمل إيه. |
| Confusion matrix | True labels are rows, predicted labels are columns. | الصفوف هي الحقيقة والأعمدة هي التوقعات؛ القطر توقعات صح. |

Face raw dimensions here: HOG 8100 + uniform LBP 10 + RGB histograms 768 = 8878.
Fingerprint raw dimensions: 8100 + 10 = 8110. Voice dimensions: 13 MFCC means,
13 MFCC standard deviations, ZCR mean/std and RMS mean/std = 30.
Actual training-only PCA retained 57, 16 and 13 dimensions respectively.

بالمصري: الأبعاد دي اتحسبت بالكود الفعلي، مش من مثال في سلايد. عدد أبعاد
PCA ممكن يتغير لو داتا التدريب اتغيرت. الـfusion عندنا 73 وبعدين 86 بعد
تحويل كل جزء. بنوحد حجم مساهمة الأجزاء قبل الجمع عشان طول متجه الوش ميغلبش الصوت.

## Why Split Before StandardScaler and PCA?

The scaler learns means/standard deviations. PCA learns directions/variances.
If either sees test samples during fitting, the representation already knows
something about the final exam. This is leakage even though PCA does not use
identity labels. Fit both using training only; transform validation/test using
the fixed transforms. Validation chooses models and thresholds. Test is the
untouched final measurement, never a reason to change a threshold or demo sample.

بالمصري: الـScaler بيتعلم المتوسط والتشتت، والـPCA بيتعلم اتجاهات التباين.
لو شافوا الاختبار وقت الـfit يبقى عرفوا جزء من الامتحان حتى من غير أسماء
الأشخاص. ده تسريب بيانات. نقسم الأول، نعمل fit على التدريب بس، ونطبق نفس
التحويل على الـvalidation والـtest. نختار الموديل والحد بالـvalidation،
والـtest نقيس عليه في الآخر من غير تعديل عشان يعجبنا الرقم.

## FAR, FRR and Other Metrics

- FAR = false accepts / impostor attempts. Example: 1 false accept in 20 wrong
  claims means FAR 0.05. This is an explanatory arithmetic example, not our score.
- FRR = false rejects / genuine attempts. A correctly classified sample may
  still be rejected because it lies beyond the distance threshold.
- Accuracy: fraction of correct decisions. Precision: of accepted positives,
  how many were genuine? Recall: of genuine attempts, how many were accepted?
  F1: harmonic mean of precision and recall. Macro-F1 averages class-wise F1
  equally; binary verification F1 treats genuine acceptance as the positive class.
- Different claims on the same input are dependent attempts. Six test triplets
  tried against three claims are 18 claim attempts, not 18 independent captures.
- Command FAR is different: unsupported YES/NO recognized as a supported word.
  It does not measure biometric impostors. Undefined FAR/FRR denominators are
  recorded as null; zero-division classification scores use an explicit zero.

بالمصري: FAR معناه نسبة الغلط اللي قبلنا فيه ادعاء دخيل. FRR معناه نسبة
الغلط اللي رفضنا فيه صاحب الهوية الحقيقي. Precision بيسأل المقبولين فيهم
كام واحد صح، وRecall بيسأل الحقيقيين قبلنا منهم كام واحد. F1 بيوفق بين
الاتنين. لو جربنا نفس العينة بأكتر من ادعاء ده مش تصوير مستقل جديد.
وبالنسبة للأوامر، قبول كلمة مش مدعومة خطأ في التعرف على الكلمة، مش إثبات
إن حد دخيل عدى من تحقق الهوية.

## What Really Happened?

Selected command KNN accuracy was **55.56%**, not a successful general speech
recognizer. Two of six unsupported test words were treated as supported.
Fingerprint and fusion3 rejected three of six genuine attempts. The selected
hierarchy authenticated three of six genuine triplets and none of twelve wrong
claims. Both executed simulated actions used the wrong word: Employee GO became
STOP and Guest GO became STOP. Permission rules were
still correctly applied to the recognized word. These are different failure modes.

بالمصري: لازم نقول الحقيقة: دقة الكلمات حوالي 55.56%، واتنين من ست كلمات
مش مدعومة اتقبلوا غلط. البصمة وfusion التلاتة رفضوا نص المحاولات الحقيقية.
البايبلاين قبل تلاتة من ستة حقيقيين ومقبلش ولا واحد من اتناشر ادعاء غلط.
فعلي المحاكاة اللي اتنفذوا كانوا بكلمة غلط: GO اتقال عليها STOP للموظف وللضيف.
قاموس الصلاحيات
طبق القاعدة صح على الكلمة اللي الموديل قالها، بس الكلمة نفسها كانت غلط.

Exact-action scoring compares the actual executed word with the expected word,
not just execute versus deny. A review corrected the earlier Boolean-only score
and separated command calibration from identity calibration. The initial run is
archived. This reused-test rerun checks software correctness, not a fresh blind
performance estimate. We did not tune using test outcomes.

بالمصري: لازم نقارن الكلمة اللي اتنفذت بالكلمة المفروض تتنفذ، مش بس نسأل
حصل تنفيذ ولا لأ. المراجعة صلحت الحساب وفصلت معايرة الأوامر عن معايرة الهوية.
النتيجة القديمة محفوظة، وإعادة تشغيل نفس الاختبار دي مراجعة للكود مش امتحان
جديد بداتا محدش شافها قبل كده.

The fixed demo does not manufacture green results: Abdelrahman and renad stop
at fingerprint. Kareem passes all five gates, but OFF is recognized as ON and
the Employee role is denied. We did not replace files or relax thresholds.

بالمصري: عينات العرض مش مضمونة النجاح. عبد الرحمن وريناد وقفوا عند البصمة.
كريم عدى التحقق كله، لكن تسجيل OFF اتقال عليه ON فدور الموظف اترفض.
مغيرناش العينات ولا وسعنا الحد عشان نطلع عرض شكله ناجح.

## Limitations and Bonuses

No same-person cross-modal evidence, no liveness or anti-spoofing, only three
users, few fingerprint impressions and similar portraits. No true capture-session
metadata, external impostor population or deployment security audit was supplied.
Image/background and recording conditions may drive apparently high scores.
More consented independent data and a new predeclared evaluation are needed.

بالمصري: مفيش دليل إن التلات أنواع بيانات لنفس الشخص، ومفيش كشف حياة أو
مقاومة تزوير. الأشخاص قليلين والبصمات قليلة وفي صور شبه بعض. الخلفية أو
ظروف التسجيل ممكن تساعد الموديل بشكل مش هيعمم. نحتاج داتا حقيقية أكتر من
نفس الأشخاص وجلسات مختلفة وتقييم جديد متفق عليه من قبل ما نشوف نتائجه.

The speech-verification/permission integration and real fingerprint fusion were
implemented and exercised. Custom face data exists, but team-collected voice is
**NOT SATISFIED**. Camera mode is implemented; the browser required camera
permission and disabled Take Photo, so genuine webcam capture/inference remains
**UNVERIFIED**. To demonstrate it, open localhost in a permitted browser, allow
the real camera, capture a new face and run the actual model. Only the face is live.

بالمصري: ربط الصوت بالتحقق والصلاحيات اتنفذ، والبصمات الحقيقية والـfusion
اتقيموا. جزء الوش الخاص موجود، بس صوت جمعه الفريق مش متحقق لأننا استخدمنا
داتا عامة. الكاميرا موجودة في الكود، لكن المتصفح طلب إذن ومسمحش بصورة جديدة،
فمينفعش نقول إن عرض الكاميرا الحقيقي اتثبت. نفتح localhost في متصفح مسموح،
ندي الإذن لكاميرا حقيقية، نصور صورة جديدة ونشغل الموديل. الوش بس هو live.