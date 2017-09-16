rm -rf dataset && \
mkdir dataset && \
cd dataset && \
curl -sS http://cv.snu.ac.kr/research/VDSR/train_data.zip > 291.zip && \
unzip 291.zip && \
rm 291.zip
