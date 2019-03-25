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
short u, v, window, initial_frame, last_frame, u0, v0;
double img_ratio = 1.0;
std::string path_result;

// Honeybee configuration
int cores_per_bee;
int num_bees;
int max_gen;
int eta_m;
int eta_c;
float rate_alpha_e;
float rate_beta_e;
float rate_gamma_e;
float rate_alpha_r;
float rate_beta_r;
float rate_gamma_r;
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
	in >> rate_alpha_e;
	in >> trash;
	in >> rate_beta_e;
	in >> trash;
	in >> rate_gamma_e;
	in >> trash;
	in >> rate_alpha_r;
	in >> trash;
	in >> rate_beta_r;
	in >> trash;
	in >> rate_gamma_r;
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

	// AMD recomends multiples of 64
	localWorkSize[0] = max_work_group_size;
	localWorkSize[1] = 1;

	// Only 1 local work group to allow coordination
	globalWorkSize[0] = localWorkSize[0];
	globalWorkSize[1] = localWorkSize[1];

	// Get actual subpopulation size
	num_bees = max_work_group_size / cores_per_bee;
	printf("Number of bees: %d\n", num_bees);
	rate_alpha_e *= num_bees;
	rate_beta_e *= num_bees;
	rate_gamma_e *= num_bees;
	rate_alpha_e = round(rate_alpha_e);
	rate_beta_e = round(rate_beta_e);
	rate_gamma_e = round(rate_gamma_e);
	rate_alpha_r *= num_bees;
	rate_beta_r *= num_bees;
	rate_gamma_r *= num_bees;
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
	printf("Alpha e: %f\n", rate_alpha_e);
	printf("Beta e: %f\n", rate_beta_e);
	printf("Gamma e: %f\n", rate_gamma_e);
	printf("Alpha r: %f\n", rate_alpha_r);
	printf("Beta r: %f\n", rate_beta_r);
	printf("Gamma r: %f\n\n", rate_gamma_r);

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
	fscanf(ann_file, "%d %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&initial_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);
	u = round(std::min(ax, std::min(bx, std::min(cx, dx))));
	v = round(std::min(ay, std::min(by, std::min(cy, dy))));
	window = round(std::max(
		std::max(ax, std::max(bx, std::max(cx, dx))) - u,
		std::max(ay, std::max(by, std::max(cy, dy))) - v
	));

	while (fscanf(ann_file, "%d %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&last_frame, &ax, &ay, &bx, &by, &cx, &cy, &dx, &dy) != EOF) {

	}

	// Close file
	fclose(ann_file);

	std::cout << "Reading annotation file: " << "\n";
	std::cout << "u: " << u << "\n";
	std::cout << "v: " << v << "\n";
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
	const std::string& new_img_path = path_result + vid_name + '_' + vid_num_str + "_" + frame_num_str + "_" + info + ".jpg";
	imwrite(new_img_path, m);
}

// Main
int main(int argc, char** argv) {
	// Read config file
	read_config();
	
	// Read kernel file
	kernel_size = read_kernel(kernel_file, &kernel_src);

	// GPU preparation
	set_gpu();

	// Read annotation file
	read_ann();

	// Read the first frame pair
	cv::Mat frame1 = read_frame(initial_frame);
	cv::Mat frame2 = read_frame(initial_frame + 1);
	IplImage *img = NULL, *img2 = NULL, *img3 = NULL, *img4 = NULL, *img0 = NULL;

	// Resize if necessary
	// GPU has limited resources
	if (window > max_window) {
		img_ratio = (double)window / max_window;
		cv::Mat temp;
		cv::Size size(frame1.cols / img_ratio, frame1.rows / img_ratio);
		cv::resize(frame1, temp, size);
		frame1 = temp.clone();
		cv::resize(frame2, temp, size);
		frame2 = temp.clone();
		u = u / img_ratio;
		v = v / img_ratio;
		window = window / img_ratio;
	}
	cv::Mat frame3 = frame1.clone();
	cv::Mat frame4 = frame1.clone();
	cv::Mat frame0 = frame1.clone();
	u0 = u;
	v0 = v;

	// Size of frames
	short w = frame1.cols;
	short h = frame1.rows;
	printf("Reading frames:\n");
	printf("u: %d px\n", u);
	printf("v: %d px\n", v);
	printf("Window: %d px\n", window);
	printf("Width: %d px\n", w);
	printf("Height: %d px\n\n", h);

	// OpenCL device memory
	cl_mem d_frame1, d_frame2, d_frame3, d_frame4, d_frame0,
		d_rand1, d_rand2, d_rand3, d_rand4,
		d_mu_e_bees, d_mu_e_obj, d_lambda_e_bees, d_lambda_e_obj,
		d_mu_lambda_bees, d_mu_lambda_obj, d_mu_lambda_order,
		d_mu_r_bees, d_mu_r_obj, d_lambda_r_bees, d_lambda_r_obj,
		d_temp;

	// Pregenerated random numbers
	unsigned int *rand1 = (unsigned int*)std::malloc((w * h) * sizeof(int));
	double *rand2 = (double*)std::malloc((w * h) * sizeof(double));
	double *rand3 = (double*)std::malloc((w * h) * sizeof(double));
	double *rand4 = (double*)std::malloc((w * h) * sizeof(double));

	// Expected displacement
	short du = 0;
	short dv = 0;

	// Frames will be sent as char arrays
	unsigned char *f1 = (unsigned char *)frame1.data;
	unsigned char *f2 = (unsigned char *)frame2.data;
	unsigned char *f0 = (unsigned char *)frame0.data;

	// Two frames to show results
	double *f3 = (double *)std::malloc((w * h) * sizeof(double));
	double *f4 = (double *)std::malloc((w * h) * sizeof(double));

	// Time variables
	struct timeval begin;
	struct timeval end;

	// Every frame
	for (int current = initial_frame + 1; current <= last_frame; current++) {
		// Begin time count
		gettimeofday(&begin, NULL);

		// Generate new random numbers
		for (int i = 0; i < w * h; i++) {
			rand1[i] = cvRandInt(&rng);
			rand2[i] = cvRandReal(&rng);
			rand3[i] = get_beta(rand2[i], eta_c);
			rand4[i] = get_delta(rand2[i], eta_m);
		}

		// Clean frame 3 and 4 results
		for (int i = 0; i < w * h; i++) {
			f3[i] = 0;
			f4[i] = 0;
		}

		// Allocate GPU memory
		// Frames, each pixel has 3 colors
		d_frame0 = clCreateBuffer(context, CL_MEM_READ_WRITE | 
			CL_MEM_COPY_HOST_PTR, (w * h) * sizeof(char) * 3, f0, &err);
		d_frame1 = clCreateBuffer(context, CL_MEM_READ_WRITE | 
			CL_MEM_COPY_HOST_PTR, (w * h) * sizeof(char) * 3, f1, &err);
		d_frame2 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, (w * h) * sizeof(char) * 3, f2, &err);
		// Results
		d_frame3 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, w * h * sizeof(double), f3, &err);
		d_frame4 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, w * h * sizeof(double), f4, &err);
		// Random numbers
		d_rand1 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, w * h * sizeof(int), rand1, &err);
		d_rand2 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, w * h * sizeof(double), rand2, &err);
		d_rand3 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, w * h * sizeof(double), rand3, &err);
		d_rand4 = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, w * h * sizeof(double), rand4, &err);
		// Bee info, 2 components each
		d_mu_e_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_mu_e_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(double), NULL, &err);
		d_lambda_e_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_lambda_e_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(double), NULL, &err);
		d_mu_r_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_mu_r_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(double), NULL, &err);
		d_lambda_r_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float), NULL, &err);
		d_lambda_r_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * sizeof(double), NULL, &err);
		// Mu + lambda bees, double size to order using merge sort
		d_mu_lambda_bees = clCreateBuffer(context, CL_MEM_READ_WRITE,
			 num_bees * 2 * sizeof(float) * 2 * 2, NULL, &err);
		d_mu_lambda_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(double) * 2, NULL, &err);
		d_mu_lambda_order = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * 2 * sizeof(float) * 2, NULL, &err);
		// Memory for many purposes
		d_temp = clCreateBuffer(context, CL_MEM_READ_WRITE,
			num_bees * num_bees * num_bees * sizeof(short), NULL, &err);
		// Check correct allocation
		if (!d_frame0 || !d_frame1 || !d_frame2 || !d_frame3 || !d_frame4 ||
			!d_rand1 || !d_rand2 || !d_rand3 || !d_rand4 ||
			!d_mu_e_bees || !d_mu_e_obj || !d_lambda_e_bees || !d_lambda_e_obj ||
			!d_mu_lambda_bees || !d_mu_lambda_obj || !d_mu_lambda_order ||
			!d_mu_r_bees || !d_mu_r_obj || !d_lambda_r_bees || !d_lambda_r_obj ||
			!d_temp) {

			printf("Error: Failed to allocate device memory!\n");
			exit(1);
		}

		// Launch OpenCL kernel
		// Max: ~256 bytes of arguments
		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_frame1);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_frame2);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_frame3);
		err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_frame4);
		err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_frame0);
		err |= clSetKernelArg(kernel, 5, sizeof(short), (void *)&w);
		err |= clSetKernelArg(kernel, 6, sizeof(short), (void *)&h);
		err |= clSetKernelArg(kernel, 7, sizeof(short), (void *)&u);
		err |= clSetKernelArg(kernel, 8, sizeof(short), (void *)&v);
		err |= clSetKernelArg(kernel, 9, sizeof(short), (void *)&u0);
		err |= clSetKernelArg(kernel, 10, sizeof(short), (void *)&v0);
		err |= clSetKernelArg(kernel, 11, sizeof(short), (void *)&window);
		err |= clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&d_rand1);
		err |= clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *)&d_rand2);
		err |= clSetKernelArg(kernel, 14, sizeof(cl_mem), (void *)&d_rand3);
		err |= clSetKernelArg(kernel, 15, sizeof(cl_mem), (void *)&d_rand4);
		err |= clSetKernelArg(kernel, 16, sizeof(short), (void *)&max_gen);
		err |= clSetKernelArg(kernel, 17, sizeof(float), (void *)&rate_beta_e);
		err |= clSetKernelArg(kernel, 18, sizeof(float), (void *)&rate_alpha_e);
		err |= clSetKernelArg(kernel, 19, sizeof(float), (void *)&rate_gamma_e);
		err |= clSetKernelArg(kernel, 20, sizeof(float), (void *)&rate_beta_r);
		err |= clSetKernelArg(kernel, 21, sizeof(float), (void *)&rate_alpha_r);
		err |= clSetKernelArg(kernel, 22, sizeof(float), (void *)&rate_gamma_r);
		err |= clSetKernelArg(kernel, 23, sizeof(cl_mem), (void *)&d_mu_e_bees);
		err |= clSetKernelArg(kernel, 24, sizeof(cl_mem), (void *)&d_mu_e_obj);
		err |= clSetKernelArg(kernel, 25, sizeof(cl_mem), (void *)&d_lambda_e_bees);
		err |= clSetKernelArg(kernel, 26, sizeof(cl_mem), (void *)&d_lambda_e_obj);
		err |= clSetKernelArg(kernel, 27, sizeof(cl_mem), (void *)&d_mu_lambda_bees);
		err |= clSetKernelArg(kernel, 28, sizeof(cl_mem), (void *)&d_mu_lambda_obj);
		err |= clSetKernelArg(kernel, 29, sizeof(cl_mem), (void *)&d_mu_lambda_order);
		err |= clSetKernelArg(kernel, 30, sizeof(cl_mem), (void *)&d_mu_r_bees);
		err |= clSetKernelArg(kernel, 31, sizeof(cl_mem), (void *)&d_mu_r_obj);
		err |= clSetKernelArg(kernel, 32, sizeof(cl_mem), (void *)&d_lambda_r_bees);
		err |= clSetKernelArg(kernel, 33, sizeof(cl_mem), (void *)&d_lambda_r_obj);
		err |= clSetKernelArg(kernel, 34, sizeof(cl_mem), (void *)&d_temp);
		err |= clSetKernelArg(kernel, 35, sizeof(short), (void *)&du);
		err |= clSetKernelArg(kernel, 36, sizeof(short), (void *)&dv);
		err |= clSetKernelArg(kernel, 37, sizeof(int), (void *)&cores_per_bee);
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
		//double *mu_lambda_obj = (double *) std::malloc(num_bees * 2 * sizeof(double) * 2);
		//float *mu_lambda_order = (float *) std::malloc(num_bees * 2 * sizeof(float) * 2);
		//double *mu_obj = (double *) std::malloc(num_bees * sizeof(double));
		err = clEnqueueReadBuffer(commands, d_frame3, CL_TRUE, 0, 
			w * h * sizeof(double), f3, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(commands, d_frame4, CL_TRUE, 0, 
			w * h * sizeof(double), f4, 0, NULL, NULL);
		/*err |= clEnqueueReadBuffer(commands, d_mu_lambda_obj, CL_TRUE, 0, 
			num_bees * 2 * sizeof(double) * 2, mu_lambda_obj, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(commands, d_mu_lambda_order, CL_TRUE, 0, 
			num_bees * 2 * sizeof(float) * 2, mu_lambda_order, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(commands, d_mu_e_obj, CL_TRUE, 0, 
			num_bees * sizeof(double), mu_obj, 0, NULL, NULL);*/
		if (err != CL_SUCCESS) {
			printf("Error: Failed to read output array! %d\n", err);
			exit(1);
		}

		/*for (int i = 0; i < num_bees * 2; i++) {
			if (i < num_bees)
				printf("%d: %f: %f: %f\n", i, mu_lambda_order[i], mu_lambda_obj[i],
					mu_obj[i]);
			else
				printf("%d: %f: %f\n", i, mu_lambda_order[i], mu_lambda_obj[i]);
		}*/

		// Find best f3
		double best = f3[0];
		double worst = f3[0];
		int bestx = 0;
		int besty = 0;
		int worstx = 0;
		int worsty = 0;
		for (int ii = 0; ii < w - window; ii++) {
			for (int jj = 0; jj < h - window; jj++) {
				double val = f3[(ii) + (jj * w)];
				if (val > best) {
					best = val;
					bestx = ii;
					besty = jj;
				}
				if (val < worst) {
					worst = val;
					worstx = ii;
					worsty = jj;
				}
			}
		}
		printf("best value: %f (%d, %d)\n", best, bestx, besty);
		printf("worst value: %f (%d, %d)\n", worst, worstx, worsty);

		// Find best f4
		double best2 = f4[0];
		double worst2 = f4[0];
		for (int ii = 0; ii < w - window; ii++) {
			for (int jj = 0; jj < h - window; jj++) {
				double val = f4[(ii) + (jj * w)];
				if (val > best2) {
					best2 = val;
				}
				if (val < worst2) {
					worst2 = val;
				}
			}
		}
		printf("best value: %f (%d, %d)\n", best2, bestx, besty);
		printf("worst value: %f (%d, %d)\n", worst2, worstx, worsty);

		// Show all
		for (int ii = 0; ii < w; ii++) {
			for (int jj = 0; jj < h; jj++) {
				if (ii < w && jj < h) {
					double val = f3[(ii) + (jj * w)];
					double temp = (val - worst);
					temp /= (best - worst);
					temp *= 255;
					unsigned char rval = temp;
					frame3.data[(ii * 3) + (jj * 3 * w)] = rval;
					frame3.data[(ii * 3) + (jj * 3 * w) + 1] = rval;
					frame3.data[(ii * 3) + (jj * 3 * w) + 2] = rval;
				} else {
					frame3.data[(ii * 3) + (jj * 3 * w)] = 255;
					frame3.data[(ii * 3) + (jj * 3 * w) + 1] = 0;
					frame3.data[(ii * 3) + (jj * 3 * w) + 2] = 0;
				}
			}
		}
		for (int ii = 0; ii < w; ii++) {
			for (int jj = 0; jj < h; jj++) {
				if (ii < w && jj < h) {
					double val = f4[(ii) + (jj * w)];
					double temp = (val - worst2);
					temp /= (best2 - worst2);
					temp *= 255;
					unsigned char rval = temp;
					frame4.data[(ii * 3) + (jj * 3 * w)] = rval;
					frame4.data[(ii * 3) + (jj * 3 * w) + 1] = rval;
					frame4.data[(ii * 3) + (jj * 3 * w) + 2] = rval;
				} else {
					frame4.data[(ii * 3) + (jj * 3 * w)] = 255;
					frame4.data[(ii * 3) + (jj * 3 * w) + 1] = 0;
					frame4.data[(ii * 3) + (jj * 3 * w) + 2] = 0;
				}
			}
		}

		// No usefull results from search
		int ignore_frame = 0;
		if (best == 0.0) {
			bestx = u;
			besty = v;
			ignore_frame = 1;
		}

		// Show results
		cv::Mat t1 = frame1.clone();
		cv::Mat t2 = frame2.clone();
		cv::Mat t0 = frame0.clone();
		img = new IplImage(t1);
		img2 = new IplImage(t2);
		img3 = new IplImage(frame3);
		img4 = new IplImage(frame4);
		img0 = new IplImage(t0);

		cvRectangle(img0, cvPoint(u0 - 1, v0 - 1),
			cvPoint(window + u0 + 1, window + v0 + 1), CV_RGB(0,255,0), 1);
		cvRectangle(img, cvPoint(u - 1, v - 1),
			cvPoint(window + u + 1, window + v + 1), CV_RGB(0,255,0), 1);
		cvRectangle(img2, cvPoint(bestx - 1, besty - 1),
			cvPoint(window + bestx + 1, window + besty + 1), CV_RGB(0,255,0), 1);

		/*cvShowImage("frame1", img);
		cvShowImage("frame2", img2);
		cvShowImage("frame3", img3);
		cvShowImage("frame4", img4);
		cvShowImage("frame0", img0);

		int dis1 = 150;
		cv::moveWindow("frame1", 50, 50);
		cv::moveWindow("frame2", 50 + dis1 * 2, 50);
		cv::moveWindow("frame3", 50 + dis1 * 4, 50);
		cv::moveWindow("frame4", 50 + dis1 * 6, 50);
		cv::moveWindow("frame0", 50, 50 + dis1 * 2);*/

		// Update frame0 sometimes
		if (current % 1 == 0) {
			frame0 = frame1.clone();
			u0 = u;
			v0 = v;
		}

		// For next frame
		du = (bestx - u);
		dv = (besty - v);
		u = bestx;
		v = besty;

		// Read video file
		frame1 = frame2.clone();
		frame2 = read_frame(current + 1);

		// Resize if necessary
		if (img_ratio > 1) {
			cv::Mat temp;
			cv::Size size(w, h);
			cv::resize(frame2,temp,size);
			frame2 = temp.clone();
		}
		f1 = (unsigned char *)frame1.data;
		f2 = (unsigned char *)frame2.data;
		f0 = (unsigned char *)frame0.data;

		// End time
		gettimeofday(&end, NULL);
		long int beginl = begin.tv_sec * 1000 + begin.tv_usec / 1000;
		long int endl = end.tv_sec * 1000 + end.tv_usec / 1000;
		double elapsed_secs = double(endl - beginl) / 1000;
		printf("Elapsed seconds: %f\n", elapsed_secs);

		// Save result
		save_img(t1, current, "frame1");
		save_img(t2, current, "frame2");
		save_img(t0, current, "frame0");
		save_img(frame3, current, "frame3");
		save_img(frame4, current, "frame4");
		
		// Wait
		//cv::waitKey(0);
	}

	printf("Finished working\n\n");

	// Clean, never do this more than once, it kills the GPU
	// I've seen this, I've done this, you don't want this
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	clReleaseMemObject(d_frame0);
	clReleaseMemObject(d_frame1);
	clReleaseMemObject(d_frame2);
	clReleaseMemObject(d_frame3);
	clReleaseMemObject(d_frame4);
	clReleaseMemObject(d_rand1);
	clReleaseMemObject(d_rand2);
	clReleaseMemObject(d_rand3);
	clReleaseMemObject(d_rand4);
	clReleaseMemObject(d_mu_e_bees);
	clReleaseMemObject(d_mu_e_obj);
	clReleaseMemObject(d_lambda_e_bees);
	clReleaseMemObject(d_lambda_e_obj);
	clReleaseMemObject(d_mu_r_bees);
	clReleaseMemObject(d_mu_r_obj);
	clReleaseMemObject(d_lambda_r_bees);
	clReleaseMemObject(d_lambda_r_obj);

	// End
	//cv::waitKey(0);
	return 0;
}

