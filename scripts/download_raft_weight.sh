mkdir -p /workspace/weights/raft
cd /workspace/weights/raft
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
cd models
mv * ..
cd ..
rm -rf models
rm models.zip