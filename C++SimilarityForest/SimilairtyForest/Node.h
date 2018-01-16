#pragma once
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
using namespace std;
struct Similarity {
	double val;
	int index;
};
bool cmp(Similarity a, Similarity b) {
	return a.val < b.val;
}
class Node {
public:
	Node * left, *right;//分别为左右节点指针
	vector <double> p,q;//选取的两类样本，p对应0,q对应1
	double partition_value;//按照基尼指数设置的划分界限
	double prediction_value;//当前节点的预测值
	double gini_value;//节点的基尼指数值
	int depth;//节点当前深度
	Node() {
		this->depth = 1;
		gini_value = 1.0;
	}
	Node(int depth) {
		this->depth = depth;
		gini_value = 1.0;
	}
	double dot(vector <double> &a, vector <double> &b) {
		double res = 0.0;
		for (int i = 0; i<a.size(); i++)
			res += (a[i] * b[i]);
		return res;
	}
	double GiniStandard(int left_num,int right_num,double left_val,double right_val) {
		double left_pred, right_pred,left_gini,right_gini,left_prob;
		left_pred = left_val / left_num;
		right_pred = right_val / right_num;
		left_gini = 1 - left_pred*left_pred - (1 - left_pred)*(1 - left_pred);
		right_gini = 1 - right_pred*right_pred - (1 - right_pred)*(1 - right_pred);
		left_prob = 1.0*left_num / (left_num + right_num);
		return left_prob*left_gini + (1 - left_prob)*right_gini;
	}
	double find_split(vector <vector<double>> &x, vector<int> &y,string method,int p_index,int q_index) {
		vector <Similarity> sim;
		double tmp;
		double left_val;
		double right_val;
		double total_val;
		Similarity temp;
		if (method == "dot") {
			left_val = right_val =total_val= 0.0;
			for (int i = 0; i < y.size(); i++) {
				tmp = dot(x[p_index], x[i]) - dot(x[q_index], x[i]);
				temp.val = tmp;
				temp.index = i;
				sim.push_back(temp);
				total_val += y[i];
			}
			sort(sim.begin(), sim.end(),cmp);
			for (int i = 0; i < y.size() - 1; i++) {
				left_val += y[sim[i].index];
				right_val = total_val - left_val;
				tmp = GiniStandard(i + 1, y.size() - i - 1, left_val, right_val);
				if (tmp< gini_value) {
					gini_value = tmp;
					partition_value = (sim[i].val + sim[i + 1].val) / 2;
				}
			}
			return gini_value;
		}
		else {
			return -1.0;
		}
		
	}
	void fit(int axes,int max_depth, vector <vector<double>> &x, vector<int> &y,string calculate_method) {
		int pos_count = 0,p_index,q_index,best_p_index,best_q_index;
		double sum = 0.0;
		for (int i = 0; i < y.size(); i++) {
			sum += y[i];
			if (y[i] == 1)
				pos_count++;
		}
		prediction_value = sum / y.size();
		if (pos_count == y.size() || pos_count == 0)
			return;
		if (depth > max_depth)
			return;
		for(int i=0;i<axes;i++){
		  while (1) {
			  p_index = rand() % y.size();
			  q_index = rand() % y.size();
			  if (y[p_index] + y[q_index] == 1)  //说明正反例各一
				  break;
		  }
		  double tmp=find_split(x, y, calculate_method,p_index,q_index);
		  if (tmp >=0) {
			  if (this->gini_value == tmp) {
				  best_p_index = p_index;
				  best_q_index = q_index;
			  }
		  }
		  else {
			  break;
		  }
		}
		for (int i = 0; i < x[best_p_index].size(); i++) {
			p.push_back(x[best_p_index][i]);
		}
		for (int i = 0; i < x[best_q_index].size(); i++) {
			q.push_back(x[best_q_index][i]);
		}
		//开始划分
		vector < vector < double >> left_part_x, right_part_x;
		vector <int> left_part_y, right_part_y;
		for (int i = 0; i < y.size(); i++) {
			if (dot(x[i], p) - dot(x[i], q) <= partition_value) {
				left_part_x.push_back(x[i]);
				left_part_y.push_back(y[i]);
			}
			else {
				right_part_x.push_back(x[i]);
				right_part_y.push_back(y[i]);
			}
		}
		//递归子树
		left = new Node(depth + 1);
		right = new Node(depth + 1);
		(*left).fit(axes,max_depth,left_part_x,left_part_y,calculate_method);
		(*right).fit(axes, max_depth, right_part_x, right_part_y, calculate_method);
	}
	double predict(vector <double> & pred_data) {
		if (left == NULL)
			return prediction_value;
		else {
			double tmp = dot(p, pred_data) - dot(q, pred_data);
			if (tmp <= partition_value)
				return (*left).predict(pred_data);
			else
				return (*right).predict(pred_data);
		}
	}
};