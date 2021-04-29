xhost +
docker run -it --rm --gpus all\
      --network=host \
      --user "$(id -u):$(id -g)" \
      -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v /tmp:/tmp:rw --privileged \
      -v ~/.gitconfig:/etc/gitconfig \
      -v /home/emilio/Documents/Cuda:/workspace cuda_11.2_opencv_4.5.2_user:latest bash