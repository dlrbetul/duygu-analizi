%Dosyayý okutalým
filename = "TrkceTwit.xlsx";
data = readtable(filename,'TextType','String');
%verileri yuzde 10 olarak test ve egitim seklinde boluyoruz
ayirma = cvpartition(data.Duygu,'Holdout',0.1);
dataTrain = data(ayirma.training,:);
dataTest = data(ayirma.test,:);
%Boldugumuz verileri deegiskenlere atiyoruz
textDataTrain = dataTrain.Tweets;
textDataTest = dataTest.Tweets;
YTrain = dataTrain.Duygu;
YTest = dataTest.Duygu;
%sadece iki veya daha az karakteri olan verileri siliyoruz
documents = tokenizedDocument(textDataTrain);
canta = bagOfWords(documents);
canta = removeInfrequentWords(canta,2);
[canta,idx] = removeEmptyDocuments(canta);
YTrain(idx) = [];
%verileri modele atiyoruz.Modeli eðitiyoruz 
XTrain = canta.Counts;
model = fitcecoc(XTrain,YTrain,'Learners','linear');
%Dogruluk degerini hesapliyoruz.
documentsTest = tokenizedDocument(textDataTest);
XTest = encode(canta,documentsTest);
YPred = predict(model,XTest);
acc = sum(YPred == YTest)/numel(YTest);
%%Gui
fig = uifigure;
text = uitextarea(fig);
text.Position = [50 300 200 30];
%accurarcy
label2 = uilabel(fig,...
    'Position',[50 180 175 15],...
    'Text',"Accuracy = "+num2str(acc));
%tahmin labeli
label1 = uilabel(fig,...
    'Position',[50 220 175 15],...
    'Text','');
label3 = uilabel(fig,...
    'Position',[50 200 175 15],...
    'Text','---------------------');
label4 = uilabel(fig,...
    'Position',[50 340 175 30],...
    'Text','Dilara Iþýk-201713709029',...
    'FontSize', 15);
%Tahmin butonu
btnTahmin = uibutton(fig,'push',...
                     'Position',[50, 250, 100, 22],...
                     'ButtonPushedFcn', @(btnTahmin,event) analizBtnPushed(btnTahmin,text,canta,model,label1));
btnTahmin.Text = 'Duygu Analizi';
%butona basildiginda calisacak fonksiyon
function analizBtnPushed(btnTahmin,text,canta,model,label1)
%kullanicidan deger alip kelimelere ayiriyoruz sonra tahmin sonucunu
%yazdiriyoruz
str=text.Value;
documentsNew = tokenizedDocument(str);
input = encode(canta,documentsNew);
tahmin = predict(model,input);
if tahmin == 0
    label1.Text = "Sonuc:Nötr";
elseif tahmin == 1
    label1.Text = "Sonuc:Negatif";
elseif tahmin == 2
    label1.Text = "Sonuc:Pozitif";
end
end