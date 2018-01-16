#include "SimilarityForest.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#define test_time 100 //预测取平均的次数
#define max_depth 10 //最大树深
#define estimator 100 //森林中树的数目
#define similarity_method "dot" //相似度计算方式
#define feature_number 10 //数据集中特征的数目
#define test_proportion 200// 除以1000，即为测试集比例
#define axes 1//树节点上选取的采样个数
int main() {
	//初始化种子
	srand((unsigned)time(0));
	//数据结构定义
	ifstream fin("breast-cancer.txt");
	ofstream fout("pbreast-cancer.txt");
	string s;
	int res,index,correct_num,correct_time;
	double val,accuracy=0.0;
	char c;
	bool out_flag = true;
	vector <vector <double>> x,test_x,train_x;
	vector <int> y,test_y,train_y;
	//数据输入与规范化部分
	while (getline(fin,s) ){
		istringstream isin(s);
		vector <double> tmp;
		isin >> res;
		if (res == 2)res = 0;
		else res = 1;
		y.push_back(res);
		for (int i = 0; i < feature_number; i++) {
			tmp.push_back(0.0);
		}
		while (isin >> index) {
			isin >> c;
			isin >> val;
			tmp[index - 1] = val;
		}		
		x.push_back(tmp);
	}
	
	
	correct_time = 0;
	for (int t = 0; t < test_time; t++) {
		correct_num = 0;
		//划分测试集和训练集
		test_x.clear();
		test_y.clear();
		train_x.clear();
		train_y.clear();
		for (int i = 0; i < y.size(); i++) {
			int tmp = rand() % 1000;
			if (tmp <= test_proportion) {
				test_x.push_back(x[i]);
				test_y.push_back(y[i]);
			}
			else {
				train_x.push_back(x[i]);
				train_y.push_back(y[i]);
			}
			if (out_flag) {
				fout << y[i];
				for (int j = 0; j < x[i].size(); j++)
					fout << "," << x[i][j];
				fout << endl;
			}
		}
		out_flag = false;
		
		//构建森林
		SimilarityForest sf = SimilarityForest(max_depth, estimator, axes, similarity_method);
		//森林训练
		string flag = sf.fit(train_x, train_y);
		if (flag == "success") {
			for (int i = 0; i < test_y.size(); i++) {
				//预测
				int tmp = sf.predict(test_x[i]);
				if (tmp == test_y[i])
					correct_num++;
			}
			correct_time++;
			accuracy += 1.0*correct_num / test_y.size();
		}
	}
	cout << fixed<<setprecision(2)<<100.0*accuracy / correct_time << "%\n";
	system("pause");
}
