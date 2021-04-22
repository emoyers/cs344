xhost +
docker run -it --rm --gpus all\
      --network=host \
      -e "NUID=$(id -u)" -e "NGID=$(id -g)" \
      -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v /tmp:/tmp:rw --privileged \
      -v /home/t8162em/Documents/Other/Cuda:/workspace cuda_11.2_opencv_4.5.2:latest