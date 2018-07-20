[휼륭하신 분](https://github.com/hccho2/CNN-VGG16-DogCat-Classification)이 작성해두신 vgg_transfer_learning 파일인데
내거에 맞게 좀 수정하였다.
파일들의 이름을 하드코딩하지 않고 폴더를 지정해주면 자동으로 파일을 읽어온다던지 기본적인 내용만 추가하였다.

vgg16 네트워크를 사용하여 ImageNet으로 pre-train된 데이터를 가져와 가중치와 편향을 사용해 내 category로 재학습시키는 모델이다.

[vgg16.npy 파일](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)이 필요하니까 다운 받아서 프로젝트 폴더에 넣어주어야 한다.

해당 .npy 파일을 읽어 category 별로 re-train된 .npy 파일을 생성해 준다.

현재 23개의 카테고리에 약 300장의 데이터를 가지고 테스트 시에는 거의 100% accuracy가 나온다.
