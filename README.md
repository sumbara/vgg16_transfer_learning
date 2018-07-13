vgg16 네트워크를 사용하여 ImageNet으로 pre-train된 데이터를 가져와 가중치와 편향을 사용해 내 category로 재학습시키는 모델이다.

vgg16.npy 파일이 필요한데 https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
여기서 다운 받아서 프로젝트 폴더에 넣어주어야 한다.

해당 .npy 파일을 읽어 category 별로 re-train된 .npy 파일을 생성해 준다.

현재 23개의 카테고리에 약 300장의 데이터를 가지고 테스트 시에는 거의 100% accuracy가 나온다.
