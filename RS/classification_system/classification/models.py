# from django.db import models
# from django.contrib.auth.models import User
#
#
# class ClassificationRecord(models.Model):
#     """分类记录模型"""
#     CATEGORY_CHOICES = [
#         ('airplane', '飞机'), ('airport', '机场'), ('baseball_diamond', '棒球场'),
#         ('basketball_court', '篮球场'), ('beach', '海滩'), ('bridge', '桥梁'),
#         ('chaparral', '灌木丛'), ('church', '教堂'), ('circular_farmland', '圆形农田'),
#         ('cloud', '云'), ('commercial_area', '商业区'), ('dense_residential', '密集住宅区'),
#         ('desert', '沙漠'), ('forest', '森林'), ('freeway', '高速公路'),
#         ('golf_course', '高尔夫球场'), ('ground_track_field', '田径场'), ('harbor', '港口'),
#         ('industrial_area', '工业区'), ('intersection', '交叉路口'), ('island', '岛屿'),
#         ('lake', '湖泊'), ('meadow', '草地'), ('medium_residential', '中等住宅区'),
#         ('mobile_home_park', '移动房车公园'), ('mountain', '山脉'), ('overpass', '立交桥'),
#         ('palace', '宫殿'), ('parking_lot', '停车场'), ('railway', '铁路'),
#         ('railway_station', '火车站'), ('rectangular_farmland', '矩形农田'),
#         ('river', '河流'), ('roundabout', '环岛'), ('runway', '跑道'),
#         ('sea_ice', '海冰'), ('ship', '船只'), ('snowberg', '雪山'),
#         ('sparse_residential', '稀疏住宅区'), ('stadium', '体育场'),
#         ('storage_tank', '储油罐'), ('tennis_court', '网球场'),
#         ('terrace', '梯田'), ('thermal_power_station', '热电站'),
#         ('wetland', '湿地')
#     ]
#
#     user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='用户')
#     image = models.ImageField(upload_to='uploads/%Y/%m/%d/', verbose_name='图像')
#     original_filename = models.CharField(max_length=255, verbose_name='原始文件名')
#     predicted_class = models.CharField(max_length=50, choices=CATEGORY_CHOICES, verbose_name='预测类别')
#     confidence = models.FloatField(verbose_name='识别确信度')
#     processing_time = models.FloatField(verbose_name='处理时间(秒)')
#     created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
#
#     # 来源ZIP文件名（批量上传时记录）
#     source_zip_name = models.CharField(max_length=255, blank=True, null=True, verbose_name='来源ZIP文件')
#
#     class Meta:
#         db_table = 'classification_record'
#         verbose_name = '分类记录'
#         verbose_name_plural = '分类记录'
#         ordering = ['-created_at']
#
#     def __str__(self):
#         return f"{self.user.username} - {self.get_predicted_class_display()} - {self.created_at}"
#
#
# class ModelInfo(models.Model):
#     """模型信息模型"""
#     name = models.CharField(max_length=100, verbose_name='模型名称')
#     version = models.CharField(max_length=20, verbose_name='版本号')
#     accuracy = models.FloatField(verbose_name='准确率')
#     file_path = models.CharField(max_length=255, verbose_name='模型文件路径')
#     created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
#     is_active = models.BooleanField(default=True, verbose_name='是否激活')
#
#     class Meta:
#         db_table = 'model_info'
#         verbose_name = '模型信息'
#
#     def __str__(self):
#         return f"{self.name} v{self.version}"

from django.db import models
from django.contrib.auth.models import User


class ClassificationRecord(models.Model):
    """分类记录模型"""
    CATEGORY_CHOICES = [
        ('airplane', '飞机'), ('airport', '机场'), ('baseball_diamond', '棒球场'),
        ('basketball_court', '篮球场'), ('beach', '海滩'), ('bridge', '桥梁'),
        ('chaparral', '灌木丛'), ('church', '教堂'), ('circular_farmland', '圆形农田'),
        ('cloud', '云'), ('commercial_area', '商业区'), ('dense_residential', '密集住宅区'),
        ('desert', '沙漠'), ('forest', '森林'), ('freeway', '高速公路'),
        ('golf_course', '高尔夫球场'), ('ground_track_field', '田径场'), ('harbor', '港口'),
        ('industrial_area', '工业区'), ('intersection', '交叉路口'), ('island', '岛屿'),
        ('lake', '湖泊'), ('meadow', '草地'), ('medium_residential', '中等住宅区'),
        ('mobile_home_park', '移动房车公园'), ('mountain', '山脉'), ('overpass', '立交桥'),
        ('palace', '宫殿'), ('parking_lot', '停车场'), ('railway', '铁路'),
        ('railway_station', '火车站'), ('rectangular_farmland', '矩形农田'),
        ('river', '河流'), ('roundabout', '环岛'), ('runway', '跑道'),
        ('sea_ice', '海冰'), ('ship', '船只'), ('snowberg', '雪山'),
        ('sparse_residential', '稀疏住宅区'), ('stadium', '体育场'),
        ('storage_tank', '储油罐'), ('tennis_court', '网球场'),
        ('terrace', '梯田'), ('thermal_power_station', '热电站'),
        ('wetland', '湿地')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='用户')
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/', verbose_name='图像')
    original_filename = models.CharField(max_length=255, verbose_name='原始文件名')
    predicted_class = models.CharField(max_length=50, choices=CATEGORY_CHOICES, verbose_name='预测类别')
    confidence = models.FloatField(verbose_name='识别确信度')
    processing_time = models.FloatField(verbose_name='处理时间(秒)')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    # 来源ZIP文件名（批量上传时记录）
    source_zip_name = models.CharField(max_length=255, blank=True, null=True, verbose_name='来源ZIP文件')

    # 使用的模型名称（新增）
    model_name = models.CharField(max_length=50, default='CGFNet', blank=True, null=True, verbose_name='使用的模型')

    class Meta:
        db_table = 'classification_record'
        verbose_name = '分类记录'
        verbose_name_plural = '分类记录'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.get_predicted_class_display()} - {self.created_at}"


class ModelInfo(models.Model):
    """模型信息模型"""
    name = models.CharField(max_length=100, verbose_name='模型名称')
    version = models.CharField(max_length=20, verbose_name='版本号')
    accuracy = models.FloatField(verbose_name='准确率')
    file_path = models.CharField(max_length=255, verbose_name='模型文件路径')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    is_active = models.BooleanField(default=True, verbose_name='是否激活')

    class Meta:
        db_table = 'model_info'
        verbose_name = '模型信息'

    def __str__(self):
        return f"{self.name} v{self.version}"