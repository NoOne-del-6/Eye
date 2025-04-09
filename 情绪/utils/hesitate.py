import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from typing import Union, List

class Hesitate:
    def __init__(self, 
                 file_path: Union[str, None, List[str]] = None,
                 use_file: bool = True,
                 img_save: str = '/app/server/output'):
        self.file_path = file_path
        self.df = None
        self.params = None
        self.proportions_and_features = None
        self.z_scores = None
        self.mapped_scores = None
        self.hesitate_score = None
        self.total_time = None
        self.values = None
        self.option_percent = None

        if use_file:
            self.primary_process()
        
    def _load_data(self) -> None:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.df = pd.DataFrame([json.loads(line) for line in file.readlines()])
        except Exception as e:
            raise FileNotFoundError
    
    @staticmethod
    def _calculate_time_diff(series: pd.Series) -> pd.Series:
        # time_diff = series.diff().fillna(0)
        # time_diff[time_diff < 0] = 0  # 将负值设为0
        return series.shape[0]
    

    #  注释路径长度和
    @staticmethod
    def _calculate_gaze_path_length(group: pd.Series) -> float:
        x_diff = group['gaze_point_x'].diff().fillna(0)
        y_diff = group['gaze_point_y'].diff().fillna(0)
        path_length = np.sqrt(x_diff ** 2 + y_diff ** 2).sum()
        return path_length
    


    # 计算注视切换次数
    @staticmethod
    def _calculate_gaze_switches(group: pd.Series) -> int:
        switches = group['gazeCategory'].shift() != group['gazeCategory'] 
        return switches.sum() - 1



    # 该方法全面计算了用户在回答问题时的注视行为特征，包括注视时间、路径长度和切换次数，并确保比例计算的准确性。
    def _calculate_proportions_and_features(self, group: pd.Series) -> pd.Series:
        # 计算每题的特征
        selected_option = self._get_selected_option(group)
        group['gazeCategory'] = group.apply(lambda row: self._classify_gaze_object(row, selected_option), axis=1)
        total_time = group['timeStamp'].iloc[-1] - group['timeStamp'].iloc[0]  # 最后一个时间戳减去第一个时间戳

        self.total_time = total_time

        if total_time == 0:
            return pd.Series({
                '题目': 0,
                '实际选项文字部分': 0,
                '未选择选项文字部分': 0,
                '实际选项': 0,
                '未选择的选项': 0,
                '背景和其他': 0,
                '注视路径长度': 0,
                '注视切换次数': 0,
                '做题总时间': 0,
            })


        # 计算各类别的注视时间
        gaze_time_topic = self._calculate_time_diff(group[group['gazeCategory'] == '题目']['timeStamp'])
        gaze_time_actual_option_text = self._calculate_time_diff(group[group['gazeCategory'] == '实际选项文字部分']['timeStamp'])
        gaze_time_unselected_option_text = self._calculate_time_diff(group[group['gazeCategory'] == '未选择选项文字部分']['timeStamp'])
        gaze_time_actual_option = self._calculate_time_diff(group[group['gazeCategory'] == '实际选项']['timeStamp'])
        gaze_time_unselected_option = self._calculate_time_diff(group[group['gazeCategory'] == '未选择的选项']['timeStamp'])
        gaze_time_background = self._calculate_time_diff(group[group['gazeCategory'] == '背景和其他']['timeStamp'])

        # 计算总的注视时间
        total_gaze_time = (gaze_time_topic + gaze_time_actual_option_text +
                        gaze_time_unselected_option_text + gaze_time_actual_option +
                        gaze_time_unselected_option + gaze_time_background)

        if total_gaze_time == 0:
            return pd.Series({
                '题目': 0,
                '实际选项文字部分': 0,
                '未选择选项文字部分': 0,
                '实际选项': 0,
                '未选择的选项': 0,
                '背景和其他': 0,
                '注视路径长度': self._calculate_gaze_path_length(group),
                '注视切换次数': self._calculate_gaze_switches(group),
                '做题总时间': total_time,
            })

        # 计算各部分的百分比
        proportions = {
            '题目': round((gaze_time_topic / total_gaze_time) * 100, 1),
            '实际选项文字部分': round((gaze_time_actual_option_text / total_gaze_time) * 100, 1),
            '未选择选项文字部分': round((gaze_time_unselected_option_text / total_gaze_time) * 100, 1),
            '实际选项': round((gaze_time_actual_option / total_gaze_time) * 100, 1),
            '未选择的选项': round((gaze_time_unselected_option / total_gaze_time) * 100, 1),
            '背景和其他': round((gaze_time_background / total_gaze_time) * 100, 1),
            '注视路径长度': self._calculate_gaze_path_length(group),
            '注视切换次数': self._calculate_gaze_switches(group),
            '做题总时间': total_time,
        }

        # 确保各部分总和为100%
        gaze_categories = ['题目', '实际选项文字部分', '未选择选项文字部分', '实际选项', '未选择的选项', '背景和其他']
        proportions_sum = sum([proportions[key] for key in gaze_categories])

        # 调整比例，使总和为100%
        adjustment_factor = 100 / proportions_sum
        for key in gaze_categories:
            proportions[key] = round(proportions[key] * adjustment_factor, 1)

        return pd.Series(proportions)



    # 处理数据并计算特征
    def _feature_process(self) -> pd.DataFrame:
        df_filtered = self.df[self.df['topicIndex'] >= 0] # 表示只关注有效的题目索引。
        grouped = df_filtered.groupby('topicIndex', group_keys=False)
        self.option_percent = self._get_group_option_percent(grouped)
        proportions_and_features = grouped.apply(self._calculate_proportions_and_features).reset_index()
        return proportions_and_features
    

    @staticmethod
    def _get_one_option_percent(group) -> dict:
        percent_dict = {}
        option_list = list(map(lambda x: x[-1], group[(group['currentGazeObject'].str.startswith('题目面板')) & (group['currentGazeObject'] != '题目面板')]['currentGazeObject'].unique().tolist()))
        for key in option_list:
            percent_dict[key] = group[group['currentGazeObject'] == f'题目面板{key}'].shape[0]
        sum_choice = sum(list(percent_dict.values()))
        for key in option_list:
            percent_dict[key] = percent_dict[key] / sum_choice 
        return percent_dict


    def _get_group_option_percent(self, groups) -> List[dict]:
        return [self._get_one_option_percent(group) for _, group in groups]


    @staticmethod
    def _classify_gaze_object(row: pd.DataFrame, selected_option: str) -> str:
        gaze_object = row['currentGazeObject']
        if gaze_object == '题目面板':
            return '题目'
        elif gaze_object.startswith('题目面板'):
            option = gaze_object[-1]
            return '实际选项文字部分' if option == selected_option else '未选择选项文字部分'
        elif gaze_object.startswith('选项'):
            option = gaze_object[-1]
            return '实际选项' if option == selected_option else '未选择的选项'
        else:
            return '背景和其他'



    def _get_selected_option(self,group: pd.Series) -> List[str]:
        option_times = group[group['currentGazeObject'].str.startswith('选项')]['currentGazeObject'].value_counts()
        if not option_times.empty:
            return option_times.idxmax()[-1]  # 返回选项的最后一个字符，例如 'A', 'B', 'C'
        return None




    def primary_process(self) -> None:
        self._load_data()
        self.proportions_and_features = self._feature_process()



    #  通过线性回归分析，计算出两个特征（做题总时间和注视路径长度）对注视切换次数的影响权重
    def deriveWeightForOneQuestionare(self) -> None:
        x1, x2 = self.proportions_and_features['做题总时间'], self.proportions_and_features['注视路径长度']
        y = self.proportions_and_features['注视切换次数']
        X1 = sm.add_constant(x1)
        X2 = sm.add_constant(x2)
        model1 = sm.OLS(y, X1).fit()
        model2 = sm.OLS(y, X2).fit()
        return model1.params[1], model2.params[1]



    def fit(self) -> List[float]:
        self.params = self.deriveWeightForOneQuestionare()




    def _normal_distribution_mapping100(self) -> List[float]:
        # 计算均值和标准差
        values = np.array(self.values)
        mean_value = np.mean(values)
        std_value = np.std(values)

        # 计算Z-Score
        z_scores = [(value - mean_value) / std_value for value in values]
        self.z_scores = z_scores

        # 使用标准正态分布的CDF将Z-Score转换为百分位数
        percentiles = [stats.norm.cdf(z) for z in z_scores]

        # 将百分位数映射到0-100之间
        mapped_values = [round(percentile * 100) for percentile in percentiles]
        return mapped_values
    



    def count_hesitate(self) -> float:
        if self.params is None:
            self.fit()
        wt, wl = self.params
        self.values= self.proportions_and_features.apply(
            self._calculate_hesitation_score, axis=1, args=(wt, wl)
        ).tolist()
        self.mapped_scores = self._normal_distribution_mapping100()
        print("\n每一题归一化犹豫分数：",np.array(self.mapped_scores))
        self.hesitate_score = np.mean(np.array(self.mapped_scores))
        return self.hesitate_score



    @staticmethod
    def _calculate_hesitation_score(df:pd.Series, wt: float, wl: float) -> float:
        return wt * df['注视路径长度'] + wl * df['做题总时间']
    