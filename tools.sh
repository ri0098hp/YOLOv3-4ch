function usage {
  cat <<EOM
Usage: $(basename "$0") [OPTION]...
  -h Display help
  -a attached container
  -b buid from Dockerfile
  -d debug (you should build image manually first)
  -e exec into container
  -r simple run
  -s down the container
EOM
  exit 2
}


while getopts "abdersh" optKey; do
  case "$optKey" in
    a)
      docker start -a YOLOv3-okuda
      ;;
    b)
      docker build -t yolov3-okuda:latest .
      ;;
    d)
      docker run --name YOLOv3-okuda-debug --gpus all -it --shm-size=12g --ipc=host --rm \
      --mount type=bind,source="$(pwd)",target=/usr/src/app/ \
      --mount type=bind,source=${HOME}${USERPROFILE}/.netrc,target=/root/.netrc \
      yolov3-okuda:test
      ;;
    e)
      docker start -i YOLOv3-okuda
      ;;
    r)
      docker run --name YOLOv3-okuda --gpus all -it --shm-size=12g --ipc=host \
      --mount type=bind,source="$(pwd)"/data,target=/usr/src/app/data \
      --mount type=bind,source="$(pwd)"/dataset,target=/usr/src/app/dataset \
      --mount type=bind,source="$(pwd)"/weights,target=/usr/src/app/weights \
      --mount type=bind,source="$(pwd)"/share,target=/usr/src/app/share \
      --mount type=bind,source="$(pwd)"/runs,target=/usr/src/app/runs \
      --mount type=bind,source="$(pwd)"/wandb,target=/usr/src/app/wandb \
      --mount type=bind,source=${HOME}${USERPROFILE}/.netrc,target=/root/.netrc \
      yolov3-okuda
      ;;
    s)
      docker rm YOLOv3-okuda
      ;;
    '-h'|'--help'|* )
      usage
      ;;
  esac
done
