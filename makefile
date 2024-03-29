vid=test

all: clean build run convert

video: clean-video build-video run convert-video generate-video

clean:
	rm *.ppm

clean-video:
	rm -Rf video/
	rm $(vid).mp4

build-video:
	@echo "Compiling"
	mkdir -p video/
	nvcc -o main main.cu

build:
	@echo "Compiling"
	nvcc -o main main.cu -DIMAGE

run:
	@echo "Running"
	./main

convert:
	@echo "Converting frame to BMP"
	for f in *.ppm ; do \
		echo "$$f >> $${f/ppm/bmp}"; \
		ppmtobmp $$f >> $${f/ppm/bmp}; \
	done

convert-video:
	@echo "Converting frames to BMP"
	for f in video/*.ppm ; do \
		echo "$$f >> $${f/ppm/bmp}"; \
		ppmtobmp $$f >> $${f/ppm/bmp}; \
	done

generate-video:
	@echo "Converting to video using FFMPEG"
	ffmpeg -r 24 -f image2 -s 1850x1000 -i video/frame%04d.bmp -vcodec libx264 -crf 25  -pix_fmt yuv420p $(vid).mp4