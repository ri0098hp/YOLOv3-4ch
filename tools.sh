function usage {
  cat <<EOM
Usage: $(basename "$0") [OPTION]...
  -h          Display help
  -b buid Dockerfile
  -d debug
  -r simple run
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
      yolov3-okuda:test
      ;;
    e)
      docker start -i YOLOv3-okuda
      ;;
    r)
      docker run --name YOLOv3-okuda --gpus all -it --shm-size=12g --ipc=host \
      --mount type=bind,source="$(pwd)"/data,target=/usr/src/app/data \
      --mount type=bind,source="$(pwd)"/dataset,target=/usr/src/app/dataset \
      --mount type=bind,source="$(pwd)"/cfg,target=/usr/src/app/cfg \
      --mount type=bind,source="$(pwd)"/weights,target=/usr/src/app/weights \
      --mount type=bind,source="$(pwd)"/share,target=/usr/src/app/share \
      --mount type=bind,source="$(pwd)"/runs,target=/usr/src/app/runs \
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
