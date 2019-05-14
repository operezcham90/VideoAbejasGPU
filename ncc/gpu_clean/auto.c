// g++ auto.c -o auto

#include <stdlib.h>
#include <fstream>
using namespace std;

std::ofstream config_file;
string folders[14];
int videos[14];

// Main
int main(int argc, char** argv) {
	folders[0] = "01-Light";
	videos[0] = 33;
	folders[1] = "02-SurfaceCover";
	videos[1] = 15;
	folders[2] = "03-Specularity";
	videos[2] = 18;
	folders[3] = "04-Transparency";
	videos[3] = 20;
	folders[4] = "05-Shape";
	videos[4] = 24;
	folders[5] = "06-MotionSmoothness";
	videos[5] = 22;
	folders[6] = "07-MotionCoherence";
	videos[6] = 12;
	folders[7] = "08-Clutter";
	videos[7] = 15;
	folders[8] = "09-Confusion";
	videos[8] = 37;
	folders[9] = "10-LowContrast";
	videos[9] = 23;
	folders[10] = "11-Occlusion";
	videos[10] = 34;
	folders[11] = "12-MovingCamera";
	videos[11] = 22;
	folders[12] = "13-ZoomingCamera";
	videos[12] = 29;
	folders[13] = "14-LongDuration";
	videos[13] = 10;
	string alov = "path_alov:\n/home/lcd/\n\n";
	string result = "path_result:\n/home/lcd/result/";
	string img = "img_dir:\nimagedata++/\n\n";
	string ann = "ann_dir:\nalov300++_rectangleAnnotation_full/\n\n";
	string vid = "vid_name:\n";
	string num = "vid_num:\n";
	string all = "max_window:\n24.0\n\ncores_per_bee:\n4\n\nmax_gen:\n6\n\neta_m:\n25\n\neta_c:\n2\n\nrate_alpha_e:\n0.6\n\nrate_beta_e:\n0.1\n\nrate_gamma_e:\n0.3\n\nrate_alpha_r:\n0.6\n\nrate_beta_r:\n0.3\n\nrate_gamma_r:\n0.1";

	for (int i = 0; i < 14; i++) {
		for (int j = 1; j <= videos[i]; j++) {
			config_file.open("./config");
			config_file << alov;
			config_file << result << folders[i] << "_" << j << "\n\n";
			config_file << img << ann;
			config_file << vid << folders[i] << "\n\n";
			config_file << num << j << "\n\n";
			config_file << all;
			config_file.close();
			char vid_num_str[5];
			sprintf(vid_num_str, "%05d", j);
			string command = "./fitness > /home/lcd/result/dump" + folders[i] + "_" + vid_num_str;
			int ret = system(command.c_str());
		}
	}
}
