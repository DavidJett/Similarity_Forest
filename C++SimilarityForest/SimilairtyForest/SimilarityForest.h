#pragma once
#include "Node.h"
#include <vector>
#include <iostream>
using namespace std;
class SimilarityForest {
public:
	vector <Node> top;
	int max_depth;//最大树深限制
	int number;//森林中树的数量
	string calculate_method;//相似度计算方式
	int axes;//分割方向
	//构造函数
	SimilarityForest(int max_depth=10,int number=10,int axes=1,string calculate_method="dot"){
		this->max_depth = max_depth;
		this->calculate_method = calculate_method;
		this->number = number;
		this->axes = 1;
	}
	//森林训练
	string fit(vector<vector<double>> & x,vector<int> & y) {
		string result="success";
		if (x.size() != y.size())
			result = "size_match_fail";
		else{
			  for (int i = 0; i < number; i++) {
				  Node tmp = Node(1);
				  tmp.fit(axes, max_depth,x,y,"dot");
				  top.push_back(tmp);
			  }
		}
		return result;
	}
	double predict_value(vector<double> & pred_data) {
		double sum = 0.0;
		for (int i = 0; i < number; i++) {
			sum+=top[i].predict(pred_data);
		}
		sum /= number;
		return sum;
	}
	int predict(vector <double> &pred_data) {
		double tmp = predict_value(pred_data);
		if (tmp <= 0.5)
			return 0;
		else
			return 1;
	}
};
