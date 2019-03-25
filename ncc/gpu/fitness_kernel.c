#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define EPSILON 1e-6
#define RED 0
#define GREEN 1
#define BLUE 2
#define GRAY 3

// Division operator misbehaves with chars
int division(int dividend, int divisor) {
	return dividend / divisor;
}

// Get the color at point (x,y) of given frame
int get_color(int x, int y, int w, __global unsigned char* frame, int rgb) {
	unsigned char color = 0;

	unsigned char b = frame[(x * 3) + (y * w * 3)];
	unsigned char g = frame[(x * 3) + (y * w * 3) + 1];
	unsigned char r = frame[(x * 3) + (y * w * 3) + 2];

	if (rgb == GRAY)
		color = division(b, 3) + division(r, 3) + division(g, 3);
	if (rgb == RED)
		color = r;
	if (rgb == GREEN)
		color = g;
	if (rgb == BLUE)
		color = b;
	
	return color;
}

// Get a random integer from the pregenerated array
unsigned int random_int(unsigned int *rand_index, int rand_size, __global unsigned int* rand1) {
	unsigned int r = rand1[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Get a random double from the pregenerated array
double random_double(unsigned int *rand_index, int rand_size, __global double* rand2) {
	double r = rand2[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Get a random delta for polynomial mutation from the pregenerated array
double random_delta(unsigned int *rand_index, int rand_size, __global double* rand4) {
	double r = rand4[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Get a random beta for SBX from the pregenerated array
double random_beta(unsigned int *rand_index, int rand_size, __global double* rand3) {
	double r = rand3[rand_index[0] % rand_size];
	rand_index[0]++;
	return r;
}

// Euclidean distance
double ec_distance (double x1, double y1, double x2, double y2) {
	double a = x1 - x2;
	double b = y1 - y2;
	double d = sqrt(a * a + b * b);
	return d;
}

// Generate an initial random population
void initial_random_pop(
	unsigned int *rand_index,
	int rand_size,
	__global double* rand2,
	__global float* mu_bees,
	int n,
	int bee,
	float* limits) {

	// Two components per bee
	// First component, first core
	if (n == 0) {
		float a = random_double(rand_index, rand_size, rand2);
		float up = limits[0];
		float low = limits[1];
		if (up > limits[2])
			up = limits[2];
		if (low < limits[3])
			low = limits[3];
		mu_bees[bee * 2] = (a * (up - low)) + low;
	}

	// Second component, second core
	if (n == 1) {
		float a = random_double(rand_index, rand_size, rand2);
		float up = limits[4];
		float low = limits[5];
		if (up > limits[6])
			up = limits[6];
		if (low < limits[7])
			low = limits[7];
		mu_bees[bee * 2 + 1] = (a * (up - low)) + low;
	}

	// Wait for all kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Draw bees
void draw(
	__global float* bees,
	__global double* obj,
	int bee,
	int n,
	__global double* frame4,
	__global double* frame3,
	int w) {

	// The first core draws the bee
	if (n == 0) {
		int x = bees[bee * 2];
		int y = bees[bee * 2 + 1];
		frame4[x + y * w] = 1;
		frame3[x + y * w] = obj[bee];
	}

	// Wait for all kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Sobel operator: edge detection
double sobel(__global unsigned char* frame, int u, int v, int w, int rgb) {
	double Gx, Gy, res, frac, ent;
	
	Gx = get_color(u-1, v+1, w, frame, rgb) + 
		2 * get_color(u, v+1, w, frame, rgb) +
		get_color(u+1, v+1, w, frame, rgb) -
		get_color(u-1, v-1, w, frame, rgb) -
		2 * get_color(u, v-1, w, frame, rgb) -
		get_color(u+1, v-1, w, frame, rgb);

	Gy = get_color(u+1, v-1, w, frame, rgb) +
		2 * get_color(u+1, v, w, frame, rgb) +
		get_color(u+1, v+1, w, frame, rgb) -
		get_color(u-1, v-1, w, frame, rgb) -
		2 * get_color(u-1, v, w, frame, rgb) -
		get_color(u-1, v+1, w, frame, rgb);

	res = sqrt(Gx * Gx + Gy * Gy);

	ent = trunc(res);
	frac = res - ent;
	res = ent;
	if ((res >= 0) && (frac > 0.5))
		res++;

	return res;
}

// ZNCC: template comparation
double zncc(
	int u1,
	int v1,
	int u2,
	int v2,
	__global unsigned char* frame1,
	__global unsigned char* frame2,
	int maxX,
	int maxY,
	int window,
	int n,
	int bee,
	int global_id,
	__global double* frame3,
	__global double* res,
	int cores_per_bee) {

	int k, l, step;
	double sumnum, sumden1, sumden2, my_int_F1, my_int_F2, int_F1, int_F2;
	int w = maxX;

	my_int_F1 = 0.0;
	my_int_F2 = 0.0;
	
	if (cores_per_bee == 4) {
		step = window / 4;
		if (n == 0) {
			// Point in frame 1 window (absolute)
			u1 = u1 + window / 4;
			v1 = v1 + window / 4;

			// Point in frame 2 window (absolute)
			u2 = u2 + window / 4;
			v2 = v2 + window / 4;
		}
		if (n == 1) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 4) * 3;
			v1 = v1 + window / 4;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 4) * 3;
			v2 = v2 + window / 4;
		}
		if (n == 2) {
			// Point in frame 1 window (absolute)
			u1 = u1 + window / 4;
			v1 = v1 + (window / 4) * 3;

			// Point in frame 2 window (absolute)
			u2 = u2 + window / 4;
			v2 = v2 + (window / 4) * 3;
		}
		if (n == 3) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 4) * 3;
			v1 = v1 + (window / 4) * 3;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 4) * 3;
			v2 = v2 + (window / 4) * 3;
		}
	}
	if (cores_per_bee == 16) {
		step = window / 8;
		if (n == 0) {
			// Point in frame 1 window (absolute)
			u1 = u1 + window / 8;
			v1 = v1 + window / 8;

			// Point in frame 2 window (absolute)
			u2 = u2 + window / 8;
			v2 = v2 + window / 8;
		}
		if (n == 1) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 3;
			v1 = v1 + window / 8;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 3;
			v2 = v2 + window / 8;
		}
		if (n == 2) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 5;
			v1 = v1 + window / 8;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 5;
			v2 = v2 + window / 8;
		}
		if (n == 3) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 7;
			v1 = v1 + window / 8;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 7;
			v2 = v2 + window / 8;
		}
		if (n == 4) {
			// Point in frame 1 window (absolute)
			u1 = u1 + window / 8;
			v1 = v1 + (window / 8) * 3;

			// Point in frame 2 window (absolute)
			u2 = u2 + window / 8;
			v2 = v2 + (window / 8) * 3;
		}
		if (n == 5) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 3;
			v1 = v1 + (window / 8) * 3;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 3;
			v2 = v2 + (window / 8) * 3;
		}
		if (n == 6) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 5;
			v1 = v1 + (window / 8) * 3;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 5;
			v2 = v2 + (window / 8) * 3;
		}
		if (n == 7) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 7;
			v1 = v1 + (window / 8) * 3;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 7;
			v2 = v2 + (window / 8) * 3;
		}
		if (n == 8) {
			// Point in frame 1 window (absolute)
			u1 = u1 + window / 8;
			v1 = v1 + (window / 8) * 5;

			// Point in frame 2 window (absolute)
			u2 = u2 + window / 8;
			v2 = v2 + (window / 8) * 5;
		}
		if (n == 9) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 3;
			v1 = v1 + (window / 8) * 5;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 3;
			v2 = v2 + (window / 8) * 5;
		}
		if (n == 10) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 5;
			v1 = v1 + (window / 8) * 5;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 5;
			v2 = v2 + (window / 8) * 5;
		}
		if (n == 11) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 7;
			v1 = v1 + (window / 8) * 5;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 7;
			v2 = v2 + (window / 8) * 5;
		}
		if (n == 12) {
			// Point in frame 1 window (absolute)
			u1 = u1 + window / 8;
			v1 = v1 + (window / 8) * 7;

			// Point in frame 2 window (absolute)
			u2 = u2 + window / 8;
			v2 = v2 + (window / 8) * 7;
		}
		if (n == 13) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 3;
			v1 = v1 + (window / 8) * 7;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 3;
			v2 = v2 + (window / 8) * 7;
		}
		if (n == 14) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 5;
			v1 = v1 + (window / 8) * 7;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 5;
			v2 = v2 + (window / 8) * 7;
		}
		if (n == 15) {
			// Point in frame 1 window (absolute)
			u1 = u1 + (window / 8) * 7;
			v1 = v1 + (window / 8) * 7;

			// Point in frame 2 window (absolute)
			u2 = u2 + (window / 8) * 7;
			v2 = v2 + (window / 8) * 7;
		}
	}	
	
	for (k = -step; k <= step; k++) {
		for (l = -step; l <= step; l++) {
			int_F1 = sobel(frame1, u1 + l, v1 + k, w, RED) +
				sobel(frame1, u1 + l, v1 + k, w, BLUE) + 
				sobel(frame1, u1 + l, v1 + k, w, GREEN);
			int_F2 = sobel(frame2, u2 + l, v2 + k, w, RED) +
				sobel(frame2, u2 + l, v2 + k, w, BLUE) +
				sobel(frame2, u2 + l, v2 + k, w, GREEN);
			my_int_F1 += int_F1;
			my_int_F2 += int_F2;
		}
	}

	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// Save result in global memory to get accumulated
	frame3[global_id * 2] = my_int_F1;
	frame3[global_id * 2 + 1] = my_int_F2;
	
	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	sumnum = 0.0;
	sumden1 = 0.0;
	sumden2 = 0.0;
	if (n == 0) {
		my_int_F1 = 0;
		my_int_F2 = 0;
		
		// Get accumulated from all cores
		for (int i = 0; i < cores_per_bee; i++) {
			my_int_F1 += frame3[(global_id + i) * 2];
			my_int_F2 += frame3[(global_id + i) * 2 + 1];
		}
		
		my_int_F1 = my_int_F1 / (window * window);
		my_int_F2 = my_int_F2 / (window * window);
		
		// Save in global memory so all the cores can use it
		frame3[global_id * 2] = my_int_F1;
		frame3[global_id * 2 + 1] = my_int_F2;
	}

	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// Copy results from global memory
	my_int_F1 = frame3[(global_id - n) * 2];
	my_int_F2 = frame3[(global_id - n) * 2 + 1];		
	
	for (k = -step; k <= step; k++) {
		for (l = -step; l <= step; l++) {
			int_F1 = sobel(frame1, u1 + l, v1 + k, w, RED) + 
				sobel(frame1, u1 + l, v1 + k, w, BLUE) +
				sobel(frame1, u1 + l, v1 + k, w, GREEN);
			int_F2 = sobel(frame2, u2 + l, v2 + k, w, RED) +
				sobel(frame2, u2 + l, v2 + k, w, GREEN) +
				sobel(frame2, u2 + l, v2 + k, w, BLUE);
			sumnum += (int_F1 - my_int_F1) * (int_F2 - my_int_F2);
			sumden1 += (int_F1 - my_int_F1) * (int_F1 - my_int_F1);
			sumden2 += (int_F2 - my_int_F2) * (int_F2 - my_int_F2);
		}
	}

	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// Save result in global memory to get accumulated
	frame3[global_id * 3] = sumnum;
	frame3[global_id * 3 + 1] = sumden1;
	frame3[global_id * 3 + 2] = sumden2;
	
	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if (n == 0) {
		sumnum = 0;
		sumden1 = 0;
		sumden2 = 0;
		
		// Get accumulated from all cores
		for (int i = 0; i < cores_per_bee; i++) {
			sumnum += frame3[(global_id + i) * 3];
			sumden1 += frame3[(global_id + i) * 3 + 1];
			sumden2 += frame3[(global_id + i) * 3 + 2];
		}
		
		res[bee] += (sumnum / (sqrt(sumden1 * sumden2)));
	}

	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Clean global memory
	frame3[global_id * 3] = 0;
	frame3[global_id * 3 + 1] = 0;
	frame3[global_id * 3 + 2] = 0;
	
	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	return (sumnum / (sqrt(sumden1 * sumden2)));
}

// Fitness function
double fitness_function(
	int u,
	int v,
	int a,
	int b,
	int u0,
	int v0,
	int maxX,
	int maxY,
	__global unsigned char* frame0,
	__global unsigned char* frame1,
	__global unsigned char* frame2,
	int window,
	int n,
	int bee,
	int global_id,
	__global double* frame3,
	__global double* res,
	int cores_per_bee) {

	double f_12, F;
	
	// Clean
	res[bee] = 0;
	
	// Wait for all cores
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Compute ZNCC of sobel
	f_12 = zncc(u, v, a, b, frame1, frame2, maxX, maxY, window, n, bee, global_id,
		frame3, res, cores_per_bee);
	zncc(u0, v0, a, b, frame0, frame2, maxX, maxY, window, n, bee, global_id,
		frame3, res, cores_per_bee);

	if (f_12 >= 0.0) {
		F = f_12;
	} else {
		F = 0.0;
	}

	// Write output:
	return F;
}

// Evaluate population
void eval_pop(
	__global float* bees,
	__global double* obj,
	int n,
	int bee,
	int global_id,
	int w,
	int h,
	__global unsigned char* frame1,
	__global unsigned char* frame2,
	__global unsigned char* frame0,
	__global double* frame3,
	int u,
	int v,
	int u0,
	int v0,
	int window,
	float* limits,
	int cores_per_bee) {
	
	// Point in frame 2
	int a = bees[bee * 2];
	int b = bees[bee * 2 + 1];
	
	// Check limits, just in case
	if (a > limits[2])
		a = limits[2];
	if (a < limits[3])
		a = limits[3];
	if (b > limits[6])
		b = limits[6];
	if (b < limits[7])
		b = limits[7];

	// Get fitness value of point
	fitness_function(u, v, a, b, u0, v0, w, h, frame0, frame1, frame2, window, n, bee, 
		global_id, frame3, obj, cores_per_bee);

	// Wait for all cores
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Polynomial mutation of given individual
void mutation(
	__global float* bees,
	int bee,
	unsigned int *rand_index,
	int rand_size,
	__global double* rand4,
	float* limits) {

	double x, delta;
	int site;

	// Mutation for each component of the individual
	for (site = 0; site < 2; site++) {
		// Get limits, based on component
		double upper = limits[site * 4 + 2];
		double lower = limits[site * 4 + 3];

		// Get the value
		x = bees[bee * 2 + site];

		// Get delta
		delta = random_delta(rand_index, rand_size, rand4);

		// Apply mutation
		if (delta >= 0) {
			bees[bee * 2 + site] += delta * (upper - x);
		} else {
			bees[bee * 2 + site] += delta * (x - lower);
		}

		// Absolute limits
		if (bees[bee * 2 + site] < lower)
			bees[bee * 2 + site] = lower;
		if (bees[bee * 2 + site] > upper)
			bees[bee * 2 + site] = upper;
	}
}

// SBX crossover
void create_children(
	float p1,
	float p2,
	__global float *c1,
	__global float *c2,
	double low,
	double high,
	unsigned int *rand_index,
	int rand_size,
	__global double* rand3) {

	double beta = random_beta(rand_index, rand_size, rand3);

	double v2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2);
	double v1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2);

	if (v2 < low) v2 = low;
	if (v2 > high) v2 = high;
	if (v1 < low) v1 = low;
	if (v1 > high) v1 = high;

	*c2 = v2;
	*c1 = v1;
}

// Crossover of two parents generating two sons
void cross_over(
	int parent1,
	int parent2,
	int child1,
	int child2,
	__global float* mu_bees,
	__global float* lambda_bees,
	unsigned int *rand_index,
	int rand_size,
	__global double* rand3,
	float* limits) {

	int site;
	int nvar_real = 2;
	for (site = 0; site < nvar_real; site++) {
		double lower = limits[site * 4 + 3];
		double upper = limits[site * 4 + 2];

		create_children(
			mu_bees[parent1 * nvar_real + site],
			mu_bees[parent2 * nvar_real + site],
			&lambda_bees[child1 * nvar_real + site],
			&lambda_bees[child2 * nvar_real + site],
			lower,
			upper,
			rand_index,
			rand_size,
			rand3);
	}
}

// Generate new population using selection, crossover, mutation and random search
void generate_new_pop(
	__global unsigned int* rand1,
	unsigned int *rand_index,
	int rand_size,
	int n,
	double rate_alpha,
	double rate_beta,
	double rate_gamma,
	__global double* mu_obj,
	__global float* mu_bees,
	__global float* lambda_bees,
	__global double* rand2,
	__global double* rand3,
	__global double* rand4,
	int bee,
	float* limits,
	int num_bees) {

	int mate1, mate2, num_cross, num_mut, num_rand;

	// Truncate rates
	int rate_mut = rate_alpha;
	int rate_cross = rate_beta;
	int rate_rand = rate_gamma;

	// Mutation
	// bees from 0 to rate_mut - 1
	// Only core 0 of each bee
	if (bee >= 0 && bee <= rate_mut - 1 && n == 0) {
		// Tournament
		int a = random_int(rand_index, rand_size, rand1) % num_bees;
		int b = random_int(rand_index, rand_size, rand1) % num_bees;
		if (mu_obj[a] > mu_obj[b])
			mate1 = a;
		else
			mate1 = b;

		// Copy the individual
		lambda_bees[bee * 2] = mu_bees[mate1 * 2];
		lambda_bees[bee * 2 + 1] = mu_bees[mate1 * 2 + 1];

		// Polinomial Mutation
		mutation(lambda_bees, bee, rand_index, rand_size, rand4, limits);
	}

	// Crossover
	// bees from first_bee + rate_mut to first_bee + rate_mut + rate_cross - 1
	// rate_mut must be even if crossover happens since two sons are generated
	if (bee >= rate_mut && 
		bee <= rate_mut + rate_cross - 1 &&
		n == 0 &&
		bee % 2 == 0) {
	
		//Tournament
		int a = random_int(rand_index, rand_size, rand1) % num_bees;
		int b = random_int(rand_index, rand_size, rand1) % num_bees;
		int c = random_int(rand_index, rand_size, rand1) % num_bees;
		int d = random_int(rand_index, rand_size, rand1) % num_bees;
		if (mu_obj[a] > mu_obj[b])
			mate1 = a;
		else
			mate1 = b;
		if (mu_obj[c] > mu_obj[d])
			mate2 = c;
		else
			mate2 = d;

		// crossover SBX
		cross_over(mate1, mate2, bee, bee + 1, mu_bees, lambda_bees,
			rand_index, rand_size, rand3, limits);
	}

	// Random
	// bees from first_bee + rate_mut + rate_cross to
	// first_bee + rate_mut + rate_cross + rate_rand - 1
	if (bee >= rate_mut + rate_cross
		&& bee <= rate_mut + rate_cross + rate_rand - 1
		&& n == 0) {

		int nvar_real = 2;
		double lower;
		double upper;
		for (int j = 0; j < nvar_real; j++) {
			upper = limits[j * 4 + 2];
			lower = limits[j * 4 + 3];

			lambda_bees[bee * nvar_real + j] =
				(double)(random_double(rand_index, rand_size, rand2) * 
				(upper - lower) + lower);
		}
	}

	// Wait for all kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Merge mu and lambda populations
void merge_pop(
	__global double* mu_obj,
	__global double* lambda_obj,
	__global double* mu_lambda_obj,
	__global float* mu_lambda_bees,
	__global float* mu_bees,
	__global float* lambda_bees,
	int bee,
	int n,
	__global float* mu_lambda_order) {
	
	if (n == 0) {
		// Copy mu bee
		int mu_lambda_bee = bee * 2;
		mu_lambda_obj[mu_lambda_bee] = mu_obj[bee];
		mu_lambda_bees[mu_lambda_bee * 2] = mu_bees[bee * 2];
		mu_lambda_bees[mu_lambda_bee * 2 + 1] = mu_bees[bee * 2 + 1];
		mu_lambda_order[mu_lambda_bee] = mu_lambda_bee;

		// Copy lambda bee
		mu_lambda_bee++;
		mu_lambda_obj[mu_lambda_bee] = lambda_obj[bee];
		mu_lambda_bees[mu_lambda_bee * 2] = lambda_bees[bee * 2];
		mu_lambda_bees[mu_lambda_bee * 2 + 1] = lambda_bees[bee * 2 + 1];
		mu_lambda_order[mu_lambda_bee] = mu_lambda_bee;
	}

	// Wait for all kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Insertion sort to order local list by objective value
// n should be small
void insertion_sort(
	__global double* numbers,
	__global float* order,
	int n,
	int list_id,
	double sign) {

	int first_num = list_id * n;
	
	for (int i = 1; i < n; i++) {
		double x = numbers[first_num + i];
		float o = order[first_num + i];

		int j = i - 1;
		while (j >= 0 && numbers[first_num + j] * sign > x * sign) {
			numbers[first_num + j + 1] = numbers[first_num + j];
			order[first_num + j + 1] = order[first_num + j];
			j--;
		}
		numbers[first_num + j + 1] = x;
		order[first_num + j + 1] = o;
	}
}

// Merge sort
void merge_sort(
	__global double* numbers,
	__global float* order,
	int n,
	int sublist_id,
	int num_sublists,
	double sign,
	int max_sublists) {

	// The first number of both sublists
	int first_num = sublist_id * n;
	int first_num2 = (sublist_id + num_sublists / 2) * n;

	// The first number of the temp list
	int first_temp = n * max_sublists;

	int j = 0;
	int k = 0;
	for (int i = 0; i < n * num_sublists; i++) {
		// If both lists have numbers left
		// check wich one has the smaller number
		if (j < (n * num_sublists) / 2 && k < (n * num_sublists) / 2) {
			// Order by number
			if (numbers[first_num + j] * sign < numbers[first_num2 + k] * sign) {
				numbers[first_temp + first_num + i] =
					numbers[first_num + j];
				order[first_temp + first_num + i] =
					order[first_num + j];
				j++;
			} else {
				numbers[first_temp + first_num + i] =
					numbers[first_num2 + k];
				order[first_temp + first_num + i] =
					order[first_num2 + k];
				k++;
			}
		// If only one sublist has numbers left
		// copy all
		} else {
			if (j < k) {
				numbers[first_temp + first_num + i] =
					numbers[first_num + j];
				order[first_temp + first_num + i] =
					order[first_num + j];
				j++;
			} else if (k < j) {
				numbers[first_temp + first_num + i] =
					numbers[first_num2 + k];
				order[first_temp + first_num + i] =
					order[first_num2 + k];
				k++;
			}
		}
	}

	// Copy order from temp array to the original array
	for (int i = 0; i < n * num_sublists; i++) {
		numbers[first_num + i] = numbers[first_temp + first_num + i];
		order[first_num + i] = order[first_temp + first_num + i];
	}
}

// Get the best mu individuals
void best_mu(
	__global double* mu_lambda_obj,
	__global float* mu_lambda_order,
	int bee,
	__global float* mu_bees,
	__global double* mu_obj,
	__global float* mu_lambda_bees,
	int n,
	int num_bees) {

	// Sort local list
	if (n == 0) {
		insertion_sort(mu_lambda_obj, mu_lambda_order, 2, bee, -1);
	}

	// Wait for all kernels
	barrier(CLK_LOCAL_MEM_FENCE);

	int max_sublists = num_bees;
	// Sort global list
	for (int num_sublists = 2; num_sublists <= max_sublists; num_sublists *= 2) {
		if (bee % num_sublists == 0 && n == 0) {
			merge_sort(mu_lambda_obj, mu_lambda_order, 2, bee, num_sublists,
				-1, max_sublists);
		}
		// Wait for all kernels
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (n == 0) {
		// Copy the best mu bees
		// The actual index of the ith best bee
		int o = mu_lambda_order[bee];

		// Copy the ith best bee to the mu population
		mu_obj[bee] = mu_lambda_obj[bee];
		mu_bees[bee * 2] = mu_lambda_bees[o * 2];
		mu_bees[bee * 2 + 1] = mu_lambda_bees[o * 2 + 1];
	}

	// Wait for all kernels
	barrier(CLK_LOCAL_MEM_FENCE);
}

// OpenCL Kernel
__kernel void fitness(
	__global unsigned char* frame1, // First frame of the video
	__global unsigned char* frame2, // Second frame of the video
	__global double* frame3, // A frame used to display results
	__global double* frame4, // A frame used to display results
	__global unsigned char* frame0, // Some frame from tha past (before the first frame)

	short w, // width of the frames
	short h, // height of the frames
	short u, // position of the object in the first frame
	short v, // position of the object in the first frame
	short u0, // position of the object in frame0
	short v0, // position of the object in frame0
	short window, // size of the square window

	__global unsigned int* rand1, // random integers
	__global double* rand2, // random doubles
	__global double* rand3, // random beta for SBX crossover
	__global double* rand4, // random delta for Polynomial mutation

	short max_gen, // Number of generations for ES
	float rate_beta_e, // Percentage of crossover sons for exploration
	float rate_alpha_e, // Percentage of mutation sons for exploration
	float rate_gamma_e, // Percentage of random sons for exploration
	float rate_beta_r, // Percentage of crossover sons for foraging
	float rate_alpha_r, // Percentage of mutation sons for foraging
	float rate_gamma_r, // Percentage of random sons for foraging

	__global float* mu_e_bees, // The real value components for each parent explorer bee
	__global double* mu_e_obj, // The objective value for each parent explorer bee

	__global float* lambda_e_bees, // The real value components for each offspring explorer bee
	__global double* lambda_e_obj, // The objective value for each offspring explorer bee

	__global float* mu_lambda_bees, // The real value components for each explorer bee
	__global double* mu_lambda_obj, // The objective value for each explorer bee
	__global float* mu_lambda_order, // The order of each explorer bee

	__global float* mu_r_bees, // The real value components for each parent foraging bee
	__global double* mu_r_obj, // The objective value for each parent foraging bee

	__global float* lambda_r_bees, // The real value components for each offspring foraging bee
	__global double* lambda_r_obj, // The objective value for each offspring foraging bee

	__global short* recruiter, // designated recruiter bee for each core, extra memory is used

	short du, // Expected displacement
	short dv, // Expected displacement

	int cores_per_bee // Expected displacement
	) {
	
	// Get the absolute global id and number of cores
	int global_id = get_global_id(0) + (get_global_id(1) * get_global_size(0));
	int cores = get_global_size(0) * get_global_size(1);
	
	// Get the size of the random arrays
	int rand_size = w * h;

	// Get a random index, different for each process
	// random numbers will follow different sequences
	unsigned int rand_index = rand1[global_id % rand_size];

	// This core belongs to this bee
	int bee = global_id / cores_per_bee;

	// This is the nth core of said bee
	int n = global_id - (cores_per_bee * bee);

	// The total number of bees
	int num_bees = cores / cores_per_bee;

	// Limits for each bee component
	float limits[8];

	// First component
	limits[0] = u + window / 8;
	limits[1] = u - window / 8;
	limits[2] = w - window;
	limits[3] = 0;

	// Second component
	limits[4] = v + window / 8;
	limits[5] = v - window / 8;
	limits[6] = h - window;
	limits[7] = 0;

	// EXPLORATION PHASE
	// Generate random initial exploration individuals
	initial_random_pop(&rand_index, rand_size, rand2, mu_e_bees, n, bee, 
		limits);

	for (int generation = 0; generation < max_gen; generation ++) {
		// Evaluate parent population
		eval_pop(mu_e_bees, mu_e_obj, n, bee, global_id, w, h,
			frame1, frame2, frame0, frame3, u, v, u0, v0, window, limits, cores_per_bee);
	
		// Generate lamdba population
		generate_new_pop(rand1, &rand_index, rand_size, n,
			rate_alpha_e, rate_beta_e, rate_gamma_e, mu_e_obj, 
			mu_e_bees, lambda_e_bees, rand2, rand3, rand4, 
			bee, limits, num_bees);

		// Evaluate new population
		eval_pop(lambda_e_bees, lambda_e_obj, n, bee, global_id, w, h,
			frame1, frame2, frame0, frame3, u, v, u0, v0, window, limits, cores_per_bee);

		// Mu + Lambda
		merge_pop(mu_e_obj, lambda_e_obj, mu_lambda_obj, mu_lambda_bees, mu_e_bees,
			lambda_e_bees, bee, n, mu_lambda_order);
	
		// Select best mu
		best_mu(mu_lambda_obj, mu_lambda_order, bee, mu_e_bees, mu_e_obj, 
			mu_lambda_bees, n, num_bees);
	}

	// RECRUITMENT PHASE
	__global short* recruits = recruiter + num_bees;
	int last_recruiter;
	int min_u;
	int max_u;
	int min_v;
	int max_v;
	if (global_id == 0) {
		double sum = 0;
		int recruited_bees = 0;

		// Get accumulated fitness value
		for (int i = 0; i < num_bees; i++) {
			// 0 or negative valoes don't contribute
			if (mu_e_obj[i] > 0)
				sum += mu_e_obj[i];
		}

		// Decide resources for each recruiter
		if (sum > 0) {
			for (int i = 0; i < num_bees; i++) {
				// 0 or negative bees have no recruits
				if (mu_e_obj[i] >= 0) {
					recruits[i] = (mu_e_obj[i] / sum) * num_bees;
					recruited_bees += recruits[i];
				} else {
					recruits[i] = 0;
				}
			}
		} else {
			// Since the last search gave no results
			// make a second normal search
			// using all cores
			recruits[0] = num_bees;
			mu_e_bees[0] = u;
			mu_e_bees[1] = v;
			for (int i = 1; i < num_bees; i++) {
				recruits[i] = 0;
			}
			recruited_bees = num_bees;
		}
		
		// All cores should have some work
		// Give the difference to the best explorer bee
		if (recruited_bees < num_bees) {
			recruits[0] = recruits[0] + (num_bees - recruited_bees);
		}
		
		// Assign bees
		int current_recruiter = 0;
		for (int i = 0; i < num_bees; i++) {
			recruiter[i] = current_recruiter;
			recruits[current_recruiter]--;
			if (recruits[current_recruiter] <= 0)
				current_recruiter++;
		}
		
		// Count the real recruited bees
		for (int i = 0; i < num_bees; i++) {
			recruits[i] = 0;
		}
		for (int i = 0; i < num_bees; i++) {
			recruits[recruiter[i]]++;
		}
		
		// Get new boundaries
		min_u = mu_e_bees[0];
		max_u = mu_e_bees[0];
		min_v = mu_e_bees[1];
		max_v = mu_e_bees[1];
		for (int i = 1; i < num_bees; i++) {
			if (mu_e_bees[recruiter[i] * 2] < min_u)
				min_u = mu_e_bees[recruiter[i] * 2];
			if (mu_e_bees[recruiter[i] * 2] > max_u)
				max_u = mu_e_bees[recruiter[i] * 2];
			if (mu_e_bees[recruiter[i] * 2 + 1] < min_v)
				min_v = mu_e_bees[recruiter[i] * 2 + 1];
			if (mu_e_bees[recruiter[i] * 2 + 1] > max_v)
				max_v = mu_e_bees[recruiter[i] * 2 + 1];
		}
		// Save in global memory since all cores need the info
		mu_lambda_obj[0] = min_u;
		mu_lambda_obj[1] = max_u;
		mu_lambda_obj[2] = min_v;
		mu_lambda_obj[3] = max_v;
	}
	// Wait for all kernels
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Save in local memory
	min_u = mu_lambda_obj[0];
	max_u = mu_lambda_obj[1];
	min_v = mu_lambda_obj[2];
	max_v = mu_lambda_obj[3];
	
	// FORAGING PHASE
	// New limits based on the recruiter
	// First component
	limits[0] = mu_e_bees[recruiter[bee] * 2] + 1;
	limits[1] = mu_e_bees[recruiter[bee] * 2] - 1;
	limits[2] = max_u;
	limits[3] = min_u;

	// Second component
	limits[4] = mu_e_bees[recruiter[bee] * 2 + 1] + 1;
	limits[5] = mu_e_bees[recruiter[bee] * 2 + 1] - 1;
	limits[6] = max_v;
	limits[7] = min_v;

	// Generate random initial individuals
	initial_random_pop(&rand_index, rand_size, rand2, mu_r_bees, n, bee, 
		limits);
	
	for (int generation = 0; generation < max_gen / 2; generation ++) {
		// Evaluate parent population
		eval_pop(mu_r_bees, mu_r_obj, n, bee, global_id, w, h,
			frame1, frame2, frame0, frame3, u, v, u0, v0, window, limits, cores_per_bee);
	
		// Generate lamdba population
		generate_new_pop(rand1, &rand_index, rand_size, n,
			rate_alpha_r, rate_beta_r, rate_gamma_r, mu_r_obj, 
			mu_r_bees, lambda_r_bees, rand2, rand3, rand4, 
			bee, limits, num_bees);

		// Evaluate new population
		eval_pop(lambda_r_bees, lambda_r_obj, n, bee, global_id, w, h,
			frame1, frame2, frame0, frame3, u, v, u0, v0, window, limits, cores_per_bee);

		// Mu + Lambda
		merge_pop(mu_r_obj, lambda_r_obj, mu_lambda_obj, mu_lambda_bees, mu_r_bees,
			lambda_r_bees, bee, n, mu_lambda_order);
	
		// Select best mu
		best_mu(mu_lambda_obj, mu_lambda_order, bee, mu_r_bees, mu_r_obj, 
			mu_lambda_bees, n, num_bees);
	}
	
	// Draw bees
	draw(mu_e_bees, mu_e_obj, bee, n, frame4, frame3, w);
	draw(mu_r_bees, mu_r_obj, bee, n, frame4, frame3, w);
}
