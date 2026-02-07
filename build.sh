echo "Configuring and building Thirdparty/DBow3 ..."

cd Thirdparty/DBow3
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j


cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Uncompress vocabulary ..."

#cd Vocabulary
#gunzip superpoint_voc.yml.gz
#cd ..

echo "Configuring and building sly_slam ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make rgbd_tum -j4
