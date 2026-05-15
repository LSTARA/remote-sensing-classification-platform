from django.urls import path
from . import views

app_name = 'classification'

urlpatterns = [
    # 基础功能
    path('', views.index, name='index'),
    path('classify/', views.classify_image, name='classify'),
    path('history/', views.history, name='history'),
    path('record/<int:record_id>/', views.record_detail, name='record_detail'),
    path('record/<int:record_id>/delete/', views.delete_record, name='delete_record'),
    path('statistics/', views.statistics, name='statistics'),
    path('api/classify/', views.api_classify, name='api_classify'),

    # 批量分类功能
    path('batch/', views.batch_classify, name='batch_classify'),
    path('batch/result/', views.batch_result, name='batch_result'),
    path('batch/export/', views.batch_export, name='batch_export'),
]