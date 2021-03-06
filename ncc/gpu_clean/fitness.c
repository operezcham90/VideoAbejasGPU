// To compile:
// g++ fitness.c -lOpenCL -o fitness `pkg-config --cflags --libs opencv`

// OpenCV (Computer Vision)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// OpenCL (Parallel Computing)
#include <CL/cl.h>

// Other
#include <math.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>

// GPU configuration
std::string kernel_file = "fitness_kernel.c";
std::string kernel_file_bee = "fitness_kernel_bee.c";
char *kernel_src;
long kernel_size;
int err;
cl_device_id device_id;
cl_context context;
cl_command_queue commands;
cl_program program;
cl_kernel kernel;
size_t max_work_group_size;
size_t localWorkSize[2], globalWorkSize[2];

// ALOV++ configuration
std::string path_alov;
std::string img_dir;
std::string ann_dir;
std::string vid_name;
int vid_num;
double max_window;
short u, v, window, initial_frame, last_frame, u0, v0, w0, h0;
double img_ratio = 1.0;
std::string path_result;
std::ofstream result_file;
int use_sobel = 0;
int resting = 0;
int honeybee = 1;

// Honeybee configuration
int cores_per_bee;
int num_bees;
short max_gen;
int eta_m;
int eta_c;
float rate_alpha_e;
float rate_beta_e;
float rate_gamma_e;
float rate_alpha_r;
float rate_beta_r;
float rate_gamma_r;
float _rate_alpha_e;
float _rate_beta_e;
float _rate_gamma_e;
float _rate_alpha_r;
float _rate_beta_r;
float _rate_gamma_r;
CvRNG rng = cvRNG(0xffffffff);

// Get beta for SBX crossover
double get_beta(double u, double eta_c) {
	double beta;
	u = 1 - u;
	double p = 1.0 / (eta_c + 1.0);

	if (u <= 0.5) {
		beta = pow(2.0 * u, p);
	} else {
		beta = pow((1.0 / (2.0 * (1.0 - u))), p);
	}
	return beta;
}

// Get delta for polynomial mutation
double get_delta(double u, double eta_m) {
	double delta;
	if (u <= 0.5) {
		delta = pow(2.0 * u, (1.0 / (eta_m + 1.0))) - 1.0;
	} else {
		delta = 1.0 - pow(2.0 * (1.0 - u), (1.0 / (eta_m + 1.0)));
	}
	return delta;
}

// Read config file
void read_config() {
	std::string trash;
	std::ifstream in("./config");
	in >> trash;
	in >> path_alov;
	in >> trash;
	in >> path_result;
	in >> trash;
	in >> img_dir;
	in >> trash;
	in >> ann_dir;
	in >> trash;
	in >> vid_name;
	in >> trash;
	in >> vid_num;
	in >> trash;
	in >> max_window;
	in >> trash;
	in >> cores_per_bee;
	in >> trash;
	in >> max_gen;
	in >> trash;
	in >> eta_m;
	in >> trash;
	in >> eta_c;
	in >> trash;
	in >> _rate_alpha_e;
	in >> trash;
	in >> _rate_beta_e;
	in >> trash;
	in >> _rate_gamma_e;
	in >> trash;
	in >> _rate_alpha_r;
	in >> trash;
	in >> _rate_beta_r;
	in >> trash;
	in >> _rate_gamma_r;
	in >> trash;
	in >> path_result;
	in.close();

	std::cout << "\n" << "Reading config file: " << "\n";
	std::cout << "Path ALOV++: " << path_alov << "\n";
	std::cout << "Path result: " << path_result << "\n";
	std::cout << "Image dir: " << img_dir << "\n";
	std::cout << "Annotation dir: " << ann_dir << "\n";
	std::cout << "Video name: " << vid_name << "\n";
	std::cout << "Video number: " << vid_num << "\n";
	std::cout << "Max window size: " << max_window << "\n";
	if (honeybee == 1) {
		std::cout << "Cores per bee: " << cores_per_bee << "\n";
		std::cout << "Max generation: " << max_gen << "\n";
		std::cout << "Eta for mutation: " << eta_m << "\n";
		std::cout << "Eta for crossover: " << eta_c << "\n";
		std::cout << "Alpha for exploration: " << rate_alpha_e << "\n";
		std::cout << "Beta for exploration: " << rate_beta_e << "\n";
		std::cout << "Gamma for exploration: " << rate_gamma_e << "\n";
		std::cout << "Alpha for foraging: " << rate_alpha_r << "\n";
		std::cout << "Beta for foraging: " << rate_beta_r << "\n";
		std::cout << "Gamma for foraging: " << rate_gamma_r << "\n";

		std::cout << "Max window / sqrt(cores per bee): " << 
		(max_window / sqrt(cores_per_bee)) << " (Should be integer)\n\n";
	}
}

// Read OpenCL kernel
long read_kernel(std::string path, char **buf) {
	FILE *fp;
	size_t fsz;
	long off_end;
	int rc;

	// Open the file
	fp = fopen(path.c_str(), "r");
	if(NULL == fp) {
		return -1L;
	}

	// Seek to the end of the file
	rc = fseek(fp, 0L, SEEK_END);
	if(0 != rc) {
		return -1L;
	}

	// Byte offset to the end of the file (size)
	if(0 > (off_end = ftell(fp))) {
		return -1L;
	}
	fsz = (size_t)off_end;

	// Allocate a buffer to hold the whole file
	*buf = (char *) malloc(fsz + 1);
		if(NULL == *buf) {
		return -1L;
	}

	// Rewind file pointer to start of file
	rewind(fp);

	// Slurp file into buffer
	if( fsz != fread(*buf, 1, fsz, fp) ) {
		free(*buf);
		return -1L;
	}

	// Close the file
	if(EOF == fclose(fp)) {
		free(*buf);
		return -1L;
	}

	// Make sure the buffer is NULL-terminated, just in case
	(*buf)[fsz] = '\0';

	// Return the file size
	return (long)fsz;
}

// GPU preparation
void set_gpu() {
	printf("GPU preparation:\n");

	// Connect to GPU device
	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);
	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0],
		gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group!\n");
		exit(1);
	}

	// Check limits
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(max_work_group_size), &max_work_group_size, NULL);
	printf("Max work group size: %lu\n", max_work_group_size);

	// Get actual subpopulation size
	num_bees = max_work_group_size / cores_per_bee;
	printf("Number of bees: %d\n", num_bees);
	rate_alpha_e = num_bees * _rate_alpha_e;
	rate_beta_e = num_bees * _rate_beta_e;
	rate_gamma_e = num_bees * _rate_gamma_e;
	rate_alpha_e = round(rate_alpha_e);
	rate_beta_e = round(rate_beta_e);
	rate_gamma_e = round(rate_gamma_e);

	rate_alpha_r = num_bees * _rate_alpha_r;
	rate_beta_r = num_bees * _rate_beta_r;
	rate_gamma_r = num_bees * _rate_gamma_r;
	rate_alpha_r = round(rate_alpha_r);
	rate_beta_r = round(rate_beta_r);
	rate_gamma_r = round(rate_gamma_r);

	// Give difference to alpha population
	if (rate_alpha_e + rate_beta_e + rate_gamma_e != num_bees) {
		rate_alpha_e += num_bees - 
			(rate_alpha_e + rate_beta_e + rate_gamma_e);
	}
	if (rate_alpha_r + rate_beta_r + rate_gamma_r != num_bees) {
		rate_alpha_r += num_bees - 
			(rate_alpha_r + rate_beta_r + rate_gamma_r);
	}

	// Beta should be even
	if ((int)rate_beta_e % 2 != 0) {
		rate_alpha_e++;
		rate_beta_e--;
	}
	if ((int)rate_beta_r % 2 != 0) {
		rate_alpha_r++;
		rate_beta_r--;
	}
	if (honeybee == 1) {
		printf("Alpha e: %f\n", rate_alpha_e);
		printf("Beta e: %f\n", rate_beta_e);
		printf("Gamma e: %f\n", rate_gamma_e);
		printf("Alpha r: %f\n", rate_alpha_r);
		printf("Beta r: %f\n", rate_beta_r);
		printf("Gamma r: %f\n\n", rate_gamma_r);
	}

	// Create a compute context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		printf("Error: Failed to create a compute context!\n");
		exit(1);
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands) {
		printf("Error: Failed to create a command commands!\n");
		exit(1);
	}

	// Create the compute program from the source file
	if(kernel_size < 0L) {
		perror("Kernel file read failed");
		exit(1);
	}
	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_src, NULL, &err);
	if (!program) {
		printf("Error: Failed to create compute program!\n");
		exit(1);
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, 
			CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "fitness", &err);
	if (!kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
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

	// Read file
	fscanf(ann_file, "%hu %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&initial_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);
	u0 = round(std::min(ax, std::min(bx, std::min(cx, dx))));
	v0 = round(std::min(ay, std::min(by, std::min(cy, dy))));
	w0 = round(std::max(ax, std::max(bx, std::max(cx, dx))) - 
		std::min(ax, std::min(bx, std::min(cx, dx))));
	h0 = round(std::max(ay, std::max(by, std::max(cy, dy))) - 
		std::min(ay, std::min(by, std::min(cy, dy))));
	window = round(std::max(
		std::max(ax, std::max(bx, std::max(cx, dx))) - u,
		std::max(ay, std::max(by, std::max(cy, dy))) - v
	));

	while (fscanf(ann_file, "%hu %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&last_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy) != EOF) {

	}

	// Close file
	fclose(ann_file);

	std::cout << "Reading annotation file: " << "\n";
	std::cout << "u0: " << u0 << "\n";
	std::cout << "v0: " << v0 << "\n";
	std::cout << "Real window: " << window << "\n";
	std::cout << "First frame: " << initial_frame << "\n";
	std::cout << "Last frame: " << last_frame << "\n\n";
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

// Save an image file
void save_img(cv::Mat m, int frame_num_m, std::string info) {
	char vid_num_str[5];
	sprintf(vid_num_str, "%05d", vid_num);
	char frame_num_str[8];
	sprintf(frame_num_str, "%08d", frame_num_m);
	const std::string& new_img_path = path_result + vid_name + '_' + 
		vid_num_str + "_" + frame_num_str + "_" + info + ".jpg";
	imwrite(new_img_path, m);
}

float get_gray(int x, int y, cv::Mat img) {
	cv::Vec3b c = img.at<cv::Vec3b>(cv::Point(x, y));
	float r = c.val[0];
	float g = c.val[1];
	float b = c.val[2];
	float gr = (r + g + b) / 3.0f;
	return gr;
}

float get_color(int x, int y, cv::Mat img, int rgb) {
	cv::Vec3b c = img.at<cv::Vec3b>(cv::Point(x, y));
	return c.val[rgb];
}

float sobel(cv::Mat frame, int u, int v, int rgb) {
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

float sobel_rgb(int x, int y, cv::Mat img) {
	double r = sobel(img, x, y, 0);
	double g = sobel(img, x, y, 1);
	double b = sobel(img, x, y, 2);
	return r + g + b;
}

// Main
int main(int argc, char** argv) {
	// Read config file
	read_config();
	
	// Read kernel file
	if (honeybee == 0)
		kernel_size = read_kernel(kernel_file, &kernel_src);
	else
		kernel_size = read_kernel(kernel_file_bee, &kernel_src);

	// Read annotation file
	read_ann();

	// Image pair
	cv::Mat frame1;
	cv::Mat frame2;

// write file
result_file.open(path_result.c_str(), std::ofstream::app);
result_file << vid_name << " " << vid_num << "\n";
// first frame
result_file << initial_frame << "," << u0 << "," << v0 << "," << w0 << "," << h0 << "\n";
result_file.close();

double avg_time = 0;

// Every frame
for (int current = initial_frame + 1; current <= last_frame; current++) {

	printf("Current frame: %d of %d\n", current, last_frame);

	struct timeval begin;
	struct timeval end;

	// GPU preparation
	set_gpu();

	// Read images
	frame1 = read_frame(current - 1);
	frame2 = read_frame(current);

	gettimeofday(&begin, NULL);

	// Get NCC
	int mx = frame1.cols;
	int my = frame1.rows;
	int nx = w0;
	int ny = h0;
	float * gamma = 
		(float*)std::malloc(((mx - nx) * (my - ny)) * sizeof(float));
	for (int x = 0; x < (mx - nx); x++) {
		for (int y = 0; y < (my - ny); y++) {
			gamma[x + y * (mx - nx)] = 0.0f;
		}
	}

	// Pregenerated random numbers
	unsigned int *rand1 = (unsigned int*)std::malloc((mx * my) * sizeof(int));
	float *rand2 = (float*)std::malloc((mx * my) * sizeof(float));
	float *rand3 = (float*)std::malloc((mx * my) * sizeof(float));
	float *rand4 = (float*)std::malloc((mx * my) * sizeof(float));

	// Generate new random numbers
	for (int i = 0; i < mx * my; i++) {
		rand1[i] = cvRandInt(&rng);
		rand2[i] = cvRandReal(&rng);
		rand3[i] = get_beta(rand2[i], eta_c);
		rand4[i] = get_delta(rand2[i], eta_m);
	}

	// filter
	float * frame1_gray = 
		(float*)std::malloc((frame1.cols * frame1.rows) * sizeof(float));
	float * frame2_gray = 
		(float*)std::malloc((frame2.cols * frame2.rows) * sizeof(float));
	for (int x = 0; x < frame1.cols; x++) {
		for (int y = 0; y < frame1.rows; y++) {
			if (use_sobel == 0) {
				frame1_gray[x + y * frame1.cols] = get_gray(x, y, frame1);
			} else {
				frame1_gray[x + y * frame1.cols] = sobel_rgb(x, y, frame1);
			}
		}
	}
	for (int x = 0; x < frame2.cols; x++) {
		for (int y = 0; y < frame2.rows; y++) {
			if (use_sobel == 0) {
				frame2_gray[x + y * frame2.cols] = get_gray(x, y, frame2);
			} else {
				frame2_gray[x + y * frame2.cols] = sobel_rgb(x, y, frame2);
			}
		}
	}

	// AMD recomends multiples of 64
	if (honeybee == 0) {
		localWorkSize[0] = 16;
		localWorkSize[1] = 16;
	} else {
		localWorkSize[0] = max_work_group_size;
		localWorkSize[1] = 1;
	}

	// Only 1 local work group to allow coordination
	if (honeybee == 0) {
		globalWorkSize[0] = (int)(localWorkSize[0] *
			ceil((mx - nx) / (float)(localWorkSize[0])));
		globalWorkSize[1] = (int)(localWorkSize[0] *
			ceil((my - ny) / (float)(localWorkSize[0])));
	} else {
		// Only 1 local work group to allow coordination
		globalWorkSize[0] = localWorkSize[0];
		globalWorkSize[1] = localWorkSize[1];
	}

	// OpenCL device memory
	cl_mem d_frame1, d_frame2, d_gamma, d_rand1, d_rand2, d_rand3, d_rand4,
		d_mu_e_bees, d_mu_e_obj, d_lambda_e_bees, d_lambda_e_obj,
		d_mu_lambda_bees, d_mu_lambda_obj, d_mu_lambda_order,
		d_mu_r_bees, d_mu_r_obj, d_lambda_r_bees, d_lambda_r_obj, 
		d_temp;

	// Allocate GPU memory
	d_frame1 = clCreateBuffer(context, CL_MEM_READ_WRITE | 
		CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), frame1_gray, &err);
	d_frame2 = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), frame2_gray, &err);
	d_gamma = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, ((mx - nx) * (my - ny)) * sizeof(float), gamma, &err);
	if (honeybee == 1) {
		// Random numbers
		d_rand1 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(int), rand1, &err);
		d_rand2 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), rand2, &err);
		d_rand3 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), rand3, &err);
		d_rand4 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, (mx * my) * sizeof(float), rand4, &err);

		// Bees
		d_mu_e_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_mu_e_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(float), NULL, &err);
		d_lambda_e_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_lambda_e_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(float), NULL, &err);

		// Mu + lambda bees, double size to order using merge sort
		d_mu_lambda_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			 num_bees * 2 * sizeof(float) * 2 * 2, NULL, &err);
		d_mu_lambda_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float) * 2, NULL, &err);
		d_mu_lambda_order = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float) * 2, NULL, &err);

		// Bees
		d_mu_r_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_mu_r_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(float), NULL, &err);
		d_lambda_r_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_lambda_r_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(float), NULL, &err);

		d_temp = clCreateBuffer(context, CL_MEM_READ_WRITE,
			2 * num_bees * sizeof(short), NULL, &err);
	}
	// Check correct allocation
	if (!d_frame1 || !d_frame2 || !d_gamma) {
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	// Launch OpenCL kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_frame1);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_frame2);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_gamma);
	err |= clSetKernelArg(kernel, 3, sizeof(short), (short *)&u0);
	err |= clSetKernelArg(kernel, 4, sizeof(short), (short *)&v0);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&nx);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&ny);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&mx);
	err |= clSetKernelArg(kernel, 8, sizeof(int), (void *)&my);
	if (honeybee == 1) {
		err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&d_rand1);
		err |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&d_rand2);
		err |= clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&d_rand3);
		err |= clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&d_rand4);

		err |= clSetKernelArg(kernel, 13, sizeof(short), (void *)&max_gen);
		err |= clSetKernelArg(kernel, 14, sizeof(float), (void *)&rate_beta_e);
		err |= clSetKernelArg(kernel, 15, sizeof(float), (void *)&rate_alpha_e);
		err |= clSetKernelArg(kernel, 16, sizeof(float), (void *)&rate_gamma_e);
		err |= clSetKernelArg(kernel, 17, sizeof(float), (void *)&rate_beta_r);
		err |= clSetKernelArg(kernel, 18, sizeof(float), (void *)&rate_alpha_r);
		err |= clSetKernelArg(kernel, 19, sizeof(float), (void *)&rate_gamma_r);

		err |= clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *)&d_mu_e_bees);
		err |= clSetKernelArg(kernel, 21, sizeof(cl_mem), (void *)&d_mu_e_obj);
		err |= clSetKernelArg(kernel, 22, sizeof(cl_mem), (void *)&d_lambda_e_bees);
		err |= clSetKernelArg(kernel, 23, sizeof(cl_mem), (void *)&d_lambda_e_obj);

		err |= clSetKernelArg(kernel, 24, sizeof(cl_mem), (void *)&d_mu_lambda_bees);
		err |= clSetKernelArg(kernel, 25, sizeof(cl_mem), (void *)&d_mu_lambda_obj);
		err |= clSetKernelArg(kernel, 26, sizeof(cl_mem), (void *)&d_mu_lambda_order);

		err |= clSetKernelArg(kernel, 27, sizeof(cl_mem), (void *)&d_mu_r_bees);
		err |= clSetKernelArg(kernel, 28, sizeof(cl_mem), (void *)&d_mu_r_obj);
		err |= clSetKernelArg(kernel, 29, sizeof(cl_mem), (void *)&d_lambda_r_bees);
		err |= clSetKernelArg(kernel, 30, sizeof(cl_mem), (void *)&d_lambda_r_obj);

		err |= clSetKernelArg(kernel, 31, sizeof(cl_mem), (void *)&d_temp);
	}
	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, 
		globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
	clFinish(commands);

	//Retrieve result from device
	err = clEnqueueReadBuffer(commands, d_gamma, CL_TRUE, 0, 
		(mx - nx) * (my - ny) * sizeof(float), gamma, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	// find the best gamma
	float maxGamma = gamma[0];
	int maxI = 0;
	int todos = 0;
	for (int i = 1; i < (mx - nx) * (my - ny); i++) {
		if (gamma[i] > maxGamma) {
			maxGamma = gamma[i];
			maxI = i;
			if (gamma[i] != -100.00f)
				todos = 1;
		}
	}
	if (todos == 1) {
		printf("Valid result: %f\n", maxGamma);
	}

	// new u0,v0
	u0 = maxI % (mx - nx);
	v0 = maxI / (mx - nx);

	// Release memory
	clReleaseMemObject(d_frame1);
	clReleaseMemObject(d_frame2);
	clReleaseMemObject(d_gamma);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	free(frame1_gray);
	free(frame2_gray);
	free(gamma);
	if (honeybee == 1) {
		free(rand1);
		free(rand2);
		free(rand3);
		free(rand4);
		clReleaseMemObject(d_rand1);
		clReleaseMemObject(d_rand2);
		clReleaseMemObject(d_rand3);
		clReleaseMemObject(d_rand4);
		clReleaseMemObject(d_mu_e_bees);
		clReleaseMemObject(d_mu_e_obj);
		clReleaseMemObject(d_lambda_e_bees);
		clReleaseMemObject(d_lambda_e_obj);
		clReleaseMemObject(d_mu_lambda_bees);
		clReleaseMemObject(d_mu_lambda_obj);
		clReleaseMemObject(d_mu_lambda_order);
		clReleaseMemObject(d_mu_r_bees);
		clReleaseMemObject(d_mu_r_obj);
		clReleaseMemObject(d_lambda_r_bees);
		clReleaseMemObject(d_lambda_r_obj);
		clReleaseMemObject(d_temp);
	}

	gettimeofday(&end, NULL);
	long int beginl = begin.tv_sec * 1000 + begin.tv_usec / 1000;
	long int endl = end.tv_sec * 1000 + end.tv_usec / 1000;
	double elapsed_secs = double(endl - beginl) / 1000;
	printf("Elapsed seconds: %f\n", elapsed_secs);

	// Save images
	//save_img(frame2, current, "read");

result_file.open(path_result.c_str(), std::ofstream::app);
	avg_time += elapsed_secs;
	result_file << current << "," << u0 
		<< "," << v0 << "," << w0 << "," << h0 << "\n";
result_file.close();

	// rest every 100 frames
	if ((current - initial_frame) % 100 == 0 && resting == 1) {
		if (avg_time /= current - initial_frame > 1.5) {
			printf("Resting...\n");
			gettimeofday(&begin, NULL);
			while(true) {
				gettimeofday(&end, NULL);
				long int beginl = begin.tv_sec * 1000 + begin.tv_usec / 1000;
				long int endl = end.tv_sec * 1000 + end.tv_usec / 1000;
				double elapsed_secs = double(endl - beginl) / 1000;
				if (elapsed_secs > 300) {
					break;
				}
			}
		}
	}
}

	avg_time /= last_frame - (initial_frame + 1);
	printf("Average seconds: %f\n", avg_time);

	return 0;
}

