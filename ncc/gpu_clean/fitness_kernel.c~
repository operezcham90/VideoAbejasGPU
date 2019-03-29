// OpenCL Kernel
__kernel void fitness(
	__global float* frame1, // First frame of the video
	__global float* frame2, // Second frame of the video
	__global float* gamma, // A frame used to display results
	short u0,
	short v0,
	int nx,
	int ny,
	int mx,
	int my
	) {

	// Get the absolute global id and number of cores
	int global_id = get_global_id(0) + (get_global_id(1) * get_global_size(0));
	int cores = get_global_size(0) * get_global_size(1);

	int u = global_id % (mx - nx);
	int v = global_id / (mx - nx);

	if (u < (mx - nx) && v < (my - ny)) {
		float tbar = 1.0 / (float)(nx * ny);
		float sumtxy = 0.0;
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				int r_y = v0 + y;
				int r_x = u0 + x;
				sumtxy += frame1[r_x + r_y * mx];
			}
		}
		tbar *= sumtxy;

		float fbar = 1.0 / (float)(nx * ny);
		float sumfxy = 0;
		for (int x = u; x < u + nx; x++) {
			for (int y = v; y < v + ny; y++) {
				sumfxy += frame2[x + y * mx];
			}
		}
		fbar *= sumfxy;

		float sumft = 0.0;
		float sumf2 = 0.0;
		float sumt2 = 0.0;
		for (int x = u; x < u + nx; x++) {
			for (int y = v; y < v + ny; y++) {
				float fxy = frame2[x + y * mx];
				int r_y = v0 + y - v;
				int r_x = u0 + x - u;
				float txy = frame1[r_x + r_y * mx];
				float fxyerr = fxy - fbar;
				float txyerr = txy - tbar;
				sumft += fxyerr * txyerr;
				sumf2 += fxyerr * fxyerr;
				sumt2 += txyerr * txyerr;
			}
		}

		gamma[global_id] = sumft / sqrt(sumf2 * sumt2);
	}
}
