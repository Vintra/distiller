DOCKER_IMAGE_NAME=distiller:lastest
CONTAINER_NAME=distiller

DATA_DIR=

if [ "$(docker images -q $DOCKER_IMAGE_NAME 2> /dev/null)" == "" ]; then
    docker build -t $DOCKER_IMAGE_NAME -f $(pwd)/docker/Dockerfile .
fi

if [ "$(docker container -q $CONTAINER_NAME 2> /dev/null)" == "" ]; then
    docker run -it --net=host \
        --runtime nvidia --gpus all \
        -e DISPLAY=$DISPLAY \
        -v $(pwd):/host -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v /media/gemma/datasets/fiftyone:/root/fiftyone \
        --name $CONTAINER_NAME \
        $DOCKER_IMAGE_NAME
else
    docker start "$CONTAINER_NAME"
    docker exec -it -e DISPLAY=$DISPLAY "$CONTAINER_NAME" /bin/bash
fi

