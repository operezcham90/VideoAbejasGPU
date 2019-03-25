// compile: g++ ncc.cpp -o ncc `pkg-config --cflags --libs opencv`

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <string>

using namespace cv;
using namespace std;

int u0, v0, um1, vm1, w0, h0, window, real_window, initial_frame, last_frame;
int double_frame = 0;
int use_sobel = 0;
std::string path_alov = "/home/lcd/";
std::string img_dir = "imagedata++/";
std::string ann_dir = "alov300++_rectangleAnnotation_full/";
std::string vid_name = "01-Light";
int vid_num = 23;
const std::string& path_result = "/home/lcd/output_ncc";
ofstream result_file;
int max_window = 24;
//int max_window = 1000;

double get_color(int x, int y, IplImage *frame, int rgb) {
	if (x < 0)
		return 0;
	else if (x >= frame->width)
		return 0;
	else if (y < 0)
		return 0;
	else if (y >= frame->height)
		return 0;
	else
		return frame->imageData[3 * (y * frame->width + x) + rgb];
}

double sobel(IplImage *frame, int u, int v, int rgb) {
	double Gx, Gy, res, frac, ent;
	
	Gx = get_color(u-1, v+1, frame, rgb) + 
		2 * get_color(u, v+1, frame, rgb) +
		get_color(u+1, v+1, frame, rgb) -
		get_color(u-1, v-1, frame, rgb) -
		2 * get_color(u, v-1, frame, rgb) -
		get_color(u+1, v-1, frame, rgb);

	Gy = get_color(u+1, v-1, frame, rgb) +
		2 * get_color(u+1, v, frame, rgb) +
		get_color(u+1, v+1, frame, rgb) -
		get_color(u-1, v-1, frame, rgb) -
		2 * get_color(u-1, v, frame, rgb) -
		get_color(u-1, v+1, frame, rgb);

	res = sqrt(Gx * Gx + Gy * Gy);

	ent = trunc(res);
	frac = res - ent;
	res = ent;
	if ((res >= 0) && (frac > 0.5))
		res++;

	return res;
}

double get_gray(int x, int y, IplImage *img) {
	if (use_sobel == 0) {
		//printf("Checking xy: %d %d\n", x, y);
		double r = img->imageData[3 * (y * img->width + x) + 0];
		double g = img->imageData[3 * (y * img->width + x) + 1];
		double b = img->imageData[3 * (y * img->width + x) + 2];
		return (r + g + b) / 3;
	} else {
		double r = sobel(img, x, y, 0);
		double g = sobel(img, x, y, 1);
		double b = sobel(img, x, y, 2);
		return r + g + b;
	}
}

// Read annotation file
void read_ann() {
	double ax, ay, bx, by, cx, cy, dx, dy;

	// Open file
	char vid_num_str[5];
	sprintf(vid_num_str, "%05d", vid_num);
	const std::string& ann_path = path_alov + ann_dir + vid_name + '/' +
		vid_name + "_video" + vid_num_str + ".ann";
	FILE* ann_file = fopen(ann_path.c_str(), "r");

	// Read initial position
	fscanf(ann_file, "%d %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&initial_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);
	u0 = round(std::min(ax, std::min(bx, std::min(cx, dx))));
	v0 = round(std::min(ay, std::min(by, std::min(cy, dy))));
	w0 = round(std::max(ax, std::max(bx, std::max(cx, dx))) - 
		std::min(ax, std::min(bx, std::min(cx, dx))));
	h0 = round(std::max(ay, std::max(by, std::max(cy, dy))) - 
		std::min(ay, std::min(by, std::min(cy, dy))));
	um1 = u0;
	vm1 = v0;
	window = std::max(w0, h0);

	// Read final frame
	while (fscanf(ann_file, "%d %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&last_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy) != EOF) {}

	// Close file
	fclose(ann_file);
}

// Read a video frame
cv::Mat read_frame(int frame_num) {
	char vid_num_str[5];
	sprintf(vid_num_str, "%05d", vid_num);
	char frame_num_str[8];
	sprintf(frame_num_str, "%08d", frame_num);
	const std::string& vid_path = path_alov + img_dir + vid_name + '/' +
		vid_name + "_video" + vid_num_str + "/" + frame_num_str + ".jpg";
	return cv::imread(vid_path, CV_LOAD_IMAGE_COLOR);
}

// main function
int main( int argc, char** argv ) {

printf("Program started\n");

double avg_time = 0;

read_ann();

// write file
result_file.open(path_result.c_str());
result_file << vid_name << " " << vid_num << "\n";
// first frame
result_file << initial_frame << "," << u0 << "," << v0 << "," << w0 << "," << h0 << "\n";
//printf("initial u0: %d v0: %d w0: %d h0: %d\n", u0, v0, w0, h0);

// Every frame
for (int current = initial_frame + 1; current <= last_frame; current++) {

	printf("Current frame: %d\n", current);

	struct timeval begin;
	struct timeval end;

	cv::Mat buf_f_mat;
	cv::Mat buf_t_mat;
	cv::Mat buf_t0_mat;

	buf_f_mat = read_frame(current);
	buf_t_mat = read_frame(current - 1);
	if (double_frame == 1) {
		if (current == initial_frame + 1)
			buf_t0_mat = read_frame(current - 1);
		else
			buf_t0_mat = read_frame(current - 2);
	}

	double img_ratio = 1.0;

	if (window > max_window) {
		img_ratio = (double)window / (double)max_window;

		// resize
		cv::Mat temp;
		cv::Mat temp2;
		cv::Mat temp3;
		cv::Size size(buf_t_mat.cols / img_ratio, buf_t_mat.rows / img_ratio);
		cv::resize(buf_t_mat, temp, size);
		if (double_frame == 1) {
			cv::resize(buf_t0_mat, temp3, size);
		}
		cv::resize(buf_f_mat, temp2, size);

		// clonar
		buf_t_mat = temp.clone();
		buf_t0_mat = temp3.clone();
		buf_f_mat = temp2.clone();

		//u0 = u0 / img_ratio;
		//v0 = v0 / img_ratio;
		//w0 = w0 / img_ratio;
		//h0 = h0 / img_ratio;
		//um1 = um1 / img_ratio;
		//vm1 = vm1 / img_ratio;
	}

	printf("    img_ratio: %f \n", img_ratio);

	IplImage *buf_f = NULL, *buf_t = NULL, *buf_t0 = NULL;
	cv::Mat t1 = buf_f_mat.clone();
	cv::Mat t2 = buf_t_mat.clone();
	cv::Mat t3 = buf_t0_mat.clone();	
	buf_f = new IplImage(t1);
	buf_t = new IplImage(t2);
	if (double_frame == 1)
		buf_t0 = new IplImage(t3);

	//printf("    Image set\n");

	gettimeofday(&begin, NULL);

	int mx = (int)buf_f->width;
	int my = (int)buf_f->height;
	int nx;
	int ny;
	int relative_v0;
	int relative_u0;
	int relative_vm1;
	int relative_um1;
	if (current == initial_frame + 1) {
		nx = w0 / img_ratio;//window;//(int)buf_t->width;
		ny = h0 / img_ratio;//window;//(int)buf_t->height;
		relative_v0 = v0 / img_ratio;
		relative_u0 = u0 / img_ratio;
		relative_vm1 = vm1 / img_ratio;
		relative_um1 = um1 / img_ratio;
	} else {
		nx = w0 / img_ratio;//window;//(int)buf_t->width;
		ny = h0 / img_ratio;//window;//(int)buf_t->height;
		relative_v0 = v0;
		relative_u0 = u0;
		relative_vm1 = vm1;
		relative_um1 = um1;
	}

	//printf("mx: %d my: %d nx: %d ny: %d v0: %d u0: %d\n", mx, my, nx, ny, relative_v0,
	//	relative_u0);

	//printf("    Image properties set\n");

	double gamma[mx - nx][my - ny];
	float f_filtered[mx][my];
	float t_filtered[mx][my];
	float t0_filtered[mx][my];

	//printf("    Arrays set\n");

printf("    start filter:\n");

	for (int x = 0; x < mx; x++) {
		for (int y = 0; y < my; y++) {
			f_filtered[x][y] = get_gray(x, y, buf_f);
			t_filtered[x][y] = get_gray(x, y, buf_t);
			if (double_frame == 1)
				t0_filtered[x][y] = get_gray(x, y, buf_t0);
		}
	}

printf("    end filter:\n");

printf("    start NCC:\n");

// repeat for second frame
	int x = 0;
	int y = 0;
	int u = 0;
	int v = 0;

	double tbar = 1 / (double)(nx * ny);
	double sumtxy = 0;
	for (x = 0; x < nx; x++) {
		for (y = 0; y < ny; y++) {
			//double txy = cvGetReal2D(buf_t, (v0 / img_ratio) + y, (u0 / img_ratio) + x);
			int r_y = relative_v0 + y;
			int r_x = relative_u0 + x;
			//double txy = buf_t->imageData[3 * (r_y * buf_t->width + r_x) + 1]; //green
			//double txy = get_gray(r_x, r_y, buf_t);
			double txy = t_filtered[r_x][r_y];
			sumtxy += txy;
		}
	}
	tbar *= sumtxy;

	double maxGamma;
	int upos = 0;
	int vpos = 0;
	for (u = 0; u < mx - nx; u++) {
		for (v = 0; v < my - ny; v++) {
			double fbar = 1 / (double)(nx * ny);
			double sumfxy = 0;
			for (x = u; x < u + nx; x++) {
				for (y = v; y < v + ny; y++) {
					//double fxy = cvGetReal2D(buf_f, y, x);
					//double fxy = buf_f->imageData[3 * (y * buf_f->width + x) + 1];
					//double fxy = get_gray(x, y, buf_f);
					double fxy = f_filtered[x][y];
					sumfxy += fxy;
				}
			}
			fbar *= sumfxy;

			double sumft = 0;
			double sumf2 = 0;
			double sumt2 = 0;
			for (x = u; x < u + nx; x++) {
				for (y = v; y < v + ny; y++) {
					//double fxy = cvGetReal2D(buf_f, y, x);
					//double fxy = buf_f->imageData[3 * (y * buf_f->width + x) + 1];
					//double fxy = get_gray(x, y, buf_f);
					double fxy = f_filtered[x][y];
					//double txy = cvGetReal2D(buf_t, (v0 / img_ratio) + y - v, (u0 / img_ratio) + x - u);
					int r_y = relative_v0 + y - v;
					int r_x = relative_u0 + x - u;
					//double txy = buf_t->imageData[3 * (r_y * buf_t->width + r_x) + 1];
					//double txy = get_gray(r_x, r_y, buf_t);
					double txy = t_filtered[r_x][r_y];

					double fxyerr = fxy - fbar;
					double txyerr = txy - tbar;

					sumft += fxyerr * txyerr;
					sumf2 += fxyerr * fxyerr;
					sumt2 += txyerr * txyerr;
				}
			}
			
			gamma[u][v] = sumft / sqrt(sumf2 * sumt2);
			if (u == 0 && v == 0) {
				maxGamma = gamma[u][v];
				upos = u;
				vpos = v;
			} else if (gamma[u][v] > maxGamma) {
				maxGamma = gamma[u][v];
				upos = u;
				vpos = v;
			}
		}
	}
printf("    end NCC:\n");

if (double_frame == 1) {
printf("    start second NCC:\n");
// repeat
	x = 0;
	y = 0;
	u = 0;
	v = 0;

	tbar = 1 / (double)(nx * ny);
	sumtxy = 0;
	for (x = 0; x < nx; x++) {
		for (y = 0; y < ny; y++) {
			//double txy = cvGetReal2D(buf_t0, (vm1 / img_ratio) + y, (um1 / img_ratio) + x);
			int r_y = relative_vm1 + y;
			int r_x = relative_um1 + x;
			//double txy = buf_t0->imageData[3 * (r_y * buf_t0->width + r_x) + 1];
			//double txy = get_gray(r_x, r_y, buf_t0);
			double txy = t0_filtered[r_x][r_y];
			sumtxy += txy;
		}
	}
	tbar *= sumtxy;

	maxGamma = -999999;
	upos = 0;
	vpos = 0;
	for (u = 0; u < mx - nx; u++) {
		for (v = 0; v < my - ny; v++) {
			double fbar = 1 / (double)(nx * ny);
			double sumfxy = 0;
			for (x = u; x < u + nx; x++) {
				for (y = v; y < v + ny; y++) {
					//double fxy = cvGetReal2D(buf_f, y, x);
					//double fxy = buf_f->imageData[3 * (y * buf_f->width + x) + 1];
					//double fxy = get_gray(x, y, buf_f);
					double fxy = f_filtered[x][y];
					sumfxy += fxy;
				}
			}
			fbar *= sumfxy;

			double sumft = 0;
			double sumf2 = 0;
			double sumt2 = 0;
			for (x = u; x < u + nx; x++) {
				for (y = v; y < v + ny; y++) {
					//double fxy = cvGetReal2D(buf_f, y, x);
					//double fxy = buf_f->imageData[3 * (y * buf_f->width + x) + 1];
					//double fxy = get_gray(x, y, buf_f);
					double fxy = f_filtered[x][y];
					//double txy = cvGetReal2D(buf_t0, (vm1 / img_ratio) + y - v, (um1 / img_ratio) + x - u);
					int r_x = relative_um1 + x - u;
					int r_y = relative_vm1 + y - v;
					//double txy = buf_t0->imageData[3 * (r_y * buf_t0->width + r_x) + 1];
					//double txy = get_gray(r_x, r_y, buf_t0);
					double txy = t0_filtered[r_x][r_y];

					double fxyerr = fxy - fbar;
					double txyerr = txy - tbar;

					sumft += fxyerr * txyerr;
					sumf2 += fxyerr * fxyerr;
					sumt2 += txyerr * txyerr;
				}
			}
			
			gamma[u][v] += sumft / sqrt(sumf2 * sumt2);
			if (u == 0 && v == 0) {
				maxGamma = gamma[u][v];
				upos = u;
				vpos = v;
			} else if (gamma[u][v] > maxGamma) {
				maxGamma = gamma[u][v];
				upos = u;
				vpos = v;
			}
		}
	}
printf("    end second NCC:\n");
}
// repeated

	printf("maxGamma: %f, u: %d , v: %d \n", maxGamma, upos, vpos);
	/*cvRectangle(buf_f, cvPoint(upos, vpos), cvPoint(upos + nx, vpos + ny), CV_RGB(0,0,0), 1);
	cvRectangle(buf_f, 
		cvPoint(upos + 1, vpos + 1),
		cvPoint(upos + nx + 1, vpos + ny + 1),
		CV_RGB(255,255,255), 1);
	cvRectangle(buf_t, 
		cvPoint(relative_u0, relative_v0), 
		cvPoint(relative_u0 + nx, relative_v0 + ny),
		CV_RGB(0,0,0), 1);
	cvRectangle(buf_t, 
		cvPoint(relative_u0 + 1, relative_v0 + 1), 
		cvPoint(relative_u0 + nx + 1, relative_v0 + ny + 1),
		CV_RGB(255,255,255), 1);

	if (double_frame == 1) {
		cvRectangle(buf_t0, 
			cvPoint(relative_um1, relative_vm1), 
			cvPoint(relative_um1 + nx, relative_vm1 + ny), 
			CV_RGB(0,0,0), 1);
		cvRectangle(buf_t0, 
			cvPoint(relative_um1 + 1, relative_vm1 + 1), 
			cvPoint(relative_um1 + nx + 1, relative_vm1 + ny + 1),
			CV_RGB(255,255,255), 1);
	}

	cvShowImage("f", buf_f);
	cvShowImage("t", buf_t);
	if (double_frame == 1)
		cvShowImage("t0", buf_t0);*/

	gettimeofday(&end, NULL);
	long int beginl = begin.tv_sec * 1000 + begin.tv_usec / 1000;
	long int endl = end.tv_sec * 1000 + end.tv_usec / 1000;
	double elapsed_secs = double(endl - beginl) / 1000;
	printf("Elapsed seconds: %f\n", elapsed_secs);

	um1 = relative_u0;// * img_ratio;
	vm1 = relative_v0;// * img_ratio;
	u0 = upos; //* img_ratio;
	v0 = vpos; //* img_ratio;
	//w0 = nx;// * img_ratio;
	//h0 = ny;// * img_ratio;
	//window = real_window;
	avg_time += elapsed_secs;
	result_file << current << "," << (u0 * img_ratio) 
		<< "," << (v0 * img_ratio) << "," << w0 << "," << h0 << "\n";
	//printf("u0: %d v0: %d w0: %d h0: %d\n", u0, v0, w0, h0);

	//cv::waitKey(0);
}

avg_time /= last_frame - (initial_frame + 1);
printf("Average seconds: %f\n", avg_time);

result_file.close();

//cv::waitKey(0);

return 0;
}
