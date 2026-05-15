# from django.shortcuts import render, redirect
# from django.contrib.auth.decorators import login_required
# from django.contrib import messages
# from django.core.paginator import Paginator
# from django.http import JsonResponse, HttpResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# from .models import ClassificationRecord
# from .model_loader import get_classifier
# import os
# from django.conf import settings
# import zipfile
# import tempfile
# import shutil
# import csv
# from datetime import datetime
#
#
# @login_required
# def index(request):
#     recent_records = ClassificationRecord.objects.filter(user=request.user)[:10]
#     total_count = ClassificationRecord.objects.filter(user=request.user).count()
#
#     # 计算平均确信度
#     from django.db.models import Avg
#     avg_result = ClassificationRecord.objects.filter(user=request.user).aggregate(avg=Avg('confidence'))
#     avg_confidence = avg_result['avg'] if avg_result['avg'] else 0
#
#     # 统计类别数量
#     category_count = ClassificationRecord.objects.filter(user=request.user).values('predicted_class').distinct().count()
#
#     context = {
#         'recent_records': recent_records,
#         'total_count': total_count,
#         'avg_confidence': avg_confidence,
#         'category_count': category_count,
#     }
#     return render(request, 'classification/index.html', context)
#
#
# # @login_required
# # def classify_image(request):
# #     if request.method == 'POST' and request.FILES.get('image'):
# #         try:
# #             uploaded_file = request.FILES['image']
# #
# #             # 验证文件类型
# #             allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
# #             if uploaded_file.content_type not in allowed_types:
# #                 messages.error(request, '请上传JPEG或PNG格式的图片')
# #                 return redirect('classification:classify')
# #
# #             # 保存临时文件
# #             file_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
# #             os.makedirs(os.path.dirname(file_path), exist_ok=True)
# #
# #             with open(file_path, 'wb+') as destination:
# #                 for chunk in uploaded_file.chunks():
# #                     destination.write(chunk)
# #
# #             # 进行分类
# #             classifier = get_classifier()
# #             result = classifier.predict(file_path)
# #
# #             # 保存记录到数据库
# #             record = ClassificationRecord.objects.create(
# #                 user=request.user,
# #                 image=f'temp/{uploaded_file.name}',
# #                 original_filename=uploaded_file.name,
# #                 predicted_class=result['predicted_class'],
# #                 confidence=result['confidence'],
# #                 processing_time=result['processing_time']
# #             )
# #
# #             # 移动文件到正式目录
# #             final_path = os.path.join(settings.MEDIA_ROOT, f'uploads/{record.id}_{uploaded_file.name}')
# #             os.makedirs(os.path.dirname(final_path), exist_ok=True)
# #             os.rename(file_path, final_path)
# #             record.image = f'uploads/{record.id}_{uploaded_file.name}'
# #             record.save()
# #
# #             # 根据确信度添加提示
# #             confidence_warning = ""
# #             confidence_level = ""
# #
# #             if result['confidence'] < 50:
# #                 confidence_warning = "⚠️ 识别确信度较低，建议使用更清晰的图像"
# #                 confidence_level = "低"
# #             elif result['confidence'] < 70:
# #                 confidence_warning = "ℹ️ 识别确信度中等，结果仅供参考"
# #                 confidence_level = "中等"
# #             elif result['confidence'] < 85:
# #                 confidence_level = "较高"
# #             else:
# #                 confidence_level = "极高"
# #
# #             context = {
# #                 'record': record,
# #                 'result': result,
# #                 'top5': result['top5_predictions'],
# #                 'confidence_warning': confidence_warning,
# #                 'confidence_level': confidence_level
# #             }
# #             return render(request, 'classification/result.html', context)
# #
# #         except FileNotFoundError as e:
# #             messages.error(request, f'模型文件未找到: {e}')
# #             return redirect('classification:classify')
# #         except Exception as e:
# #             messages.error(request, f'分类失败: {str(e)}')
# #             return redirect('classification:classify')
# #
# #     return render(request, 'classification/classify.html')
# @login_required
# def classify_image(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         try:
#             uploaded_file = request.FILES['image']
#
#             # 获取用户选择的模型
#             model_choice = request.POST.get('model_choice', 'cgfnet')
#             print(f"用户选择模型: {model_choice}")
#
#             # 验证文件类型
#             allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
#             if uploaded_file.content_type not in allowed_types:
#                 messages.error(request, '请上传JPEG或PNG格式的图片')
#                 return redirect('classification:classify')
#
#             # 保存临时文件
#             file_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
#             os.makedirs(os.path.dirname(file_path), exist_ok=True)
#
#             with open(file_path, 'wb+') as destination:
#                 for chunk in uploaded_file.chunks():
#                     destination.write(chunk)
#
#             # 根据用户选择的模型进行分类
#             classifier = get_classifier(model_choice)
#             result = classifier.predict(file_path)
#
#             # 保存记录到数据库
#             record = ClassificationRecord.objects.create(
#                 user=request.user,
#                 image=f'temp/{uploaded_file.name}',
#                 original_filename=uploaded_file.name,
#                 predicted_class=result['predicted_class'],
#                 confidence=result['confidence'],
#                 processing_time=result['processing_time'],
#                 model_name=result.get('model_name', model_choice)  # 保存使用的模型
#             )
#
#             # 移动文件到正式目录
#             final_path = os.path.join(settings.MEDIA_ROOT, f'uploads/{record.id}_{uploaded_file.name}')
#             os.makedirs(os.path.dirname(final_path), exist_ok=True)
#             os.rename(file_path, final_path)
#             record.image = f'uploads/{record.id}_{uploaded_file.name}'
#             record.save()
#
#             # 根据确信度添加提示
#             confidence_warning = ""
#             confidence_level = ""
#
#             if result['confidence'] < 50:
#                 confidence_warning = "⚠️ 识别确信度较低，建议使用更清晰的图像"
#                 confidence_level = "低"
#             elif result['confidence'] < 70:
#                 confidence_warning = "ℹ️ 识别确信度中等，结果仅供参考"
#                 confidence_level = "中等"
#             elif result['confidence'] < 85:
#                 confidence_level = "较高"
#             else:
#                 confidence_level = "极高"
#
#             context = {
#                 'record': record,
#                 'result': result,
#                 'top5': result['top5_predictions'],
#                 'confidence_warning': confidence_warning,
#                 'confidence_level': confidence_level
#             }
#             return render(request, 'classification/result.html', context)
#
#         except FileNotFoundError as e:
#             messages.error(request, f'模型文件未找到: {e}')
#             return redirect('classification:classify')
#         except Exception as e:
#             messages.error(request, f'分类失败: {str(e)}')
#             return redirect('classification:classify')
#
#     return render(request, 'classification/classify.html')
#
#
# @login_required
# def history(request):
#     records = ClassificationRecord.objects.filter(user=request.user)
#     paginator = Paginator(records, 12)
#     page_number = request.GET.get('page', 1)
#     page_obj = paginator.get_page(page_number)
#     context = {'page_obj': page_obj, 'total_count': records.count()}
#     return render(request, 'classification/history.html', context)
#
#
# @login_required
# def record_detail(request, record_id):
#     try:
#         record = ClassificationRecord.objects.get(id=record_id, user=request.user)
#         context = {'record': record}
#         return render(request, 'classification/record_detail.html', context)
#     except ClassificationRecord.DoesNotExist:
#         messages.error(request, '记录不存在')
#         return redirect('classification:history')
#
#
# @login_required
# def delete_record(request, record_id):
#     if request.method == 'POST':
#         try:
#             record = ClassificationRecord.objects.get(id=record_id, user=request.user)
#             if record.image and os.path.exists(record.image.path):
#                 os.remove(record.image.path)
#             record.delete()
#             messages.success(request, '记录已删除')
#         except ClassificationRecord.DoesNotExist:
#             messages.error(request, '记录不存在')
#     return redirect('classification:history')
#
#
# @login_required
# def statistics(request):
#     from django.db.models import Count, Avg
#     from datetime import datetime, timedelta
#
#     stats = ClassificationRecord.objects.filter(
#         user=request.user
#     ).values('predicted_class').annotate(
#         count=Count('predicted_class')
#     ).order_by('-count')[:10]
#
#     # 获取近30天统计
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=30)
#
#     daily_stats = ClassificationRecord.objects.filter(
#         user=request.user,
#         created_at__range=[start_date, end_date]
#     ).extra({'date': "DATE(created_at)"}).values('date').annotate(
#         count=Count('id')
#     ).order_by('date')
#
#     total_count = ClassificationRecord.objects.filter(user=request.user).count()
#
#     if total_count > 0:
#         avg_confidence_result = ClassificationRecord.objects.filter(
#             user=request.user
#         ).aggregate(avg=Avg('confidence'))
#         avg_confidence = avg_confidence_result['avg'] if avg_confidence_result['avg'] is not None else 0
#     else:
#         avg_confidence = 0
#
#     context = {
#         'stats': stats,
#         'daily_stats': list(daily_stats),
#         'total_count': total_count,
#         'avg_confidence': avg_confidence
#     }
#     return render(request, 'classification/statistics.html', context)
#
#
# @csrf_exempt
# @require_http_methods(["POST"])
# @login_required
# def api_classify(request):
#     if request.FILES.get('image'):
#         try:
#             uploaded_file = request.FILES['image']
#             classifier = get_classifier()
#
#             temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
#             os.makedirs(os.path.dirname(temp_path), exist_ok=True)
#             with open(temp_path, 'wb+') as f:
#                 for chunk in uploaded_file.chunks():
#                     f.write(chunk)
#
#             result = classifier.predict(temp_path)
#             os.remove(temp_path)
#
#             return JsonResponse({
#                 'success': True,
#                 'predicted_class': result['predicted_class'],
#                 'confidence': result['confidence'],
#                 'processing_time': result['processing_time']
#             })
#         except Exception as e:
#             return JsonResponse({'success': False, 'error': str(e)})
#
#     return JsonResponse({'success': False, 'error': '未提供图像'})
#
#
# # ==================== 批量分类功能 ====================
#
# def process_single_image(user, uploaded_file, source_zip_name=None):
#     """处理单张图片
#
#     Args:
#         user: 用户对象
#         uploaded_file: 上传的文件
#         source_zip_name: 来源ZIP文件名（可选）
#     """
#     try:
#         # 保存临时文件
#         temp_dir = tempfile.mkdtemp()
#         file_path = os.path.join(temp_dir, uploaded_file.name)
#
#         with open(file_path, 'wb+') as destination:
#             for chunk in uploaded_file.chunks():
#                 destination.write(chunk)
#
#         # 进行分类
#         classifier = get_classifier()
#         result = classifier.predict(file_path)
#
#         # 保存记录到数据库
#         record = ClassificationRecord.objects.create(
#             user=user,
#             image='',  # 临时，稍后更新
#             original_filename=uploaded_file.name,
#             predicted_class=result['predicted_class'],
#             confidence=result['confidence'],
#             processing_time=result['processing_time'],
#             source_zip_name=source_zip_name  # 保存来源ZIP名
#         )
#
#         # 移动文件到正式目录
#         final_filename = f"{record.id}_{uploaded_file.name}"
#         final_path = os.path.join(settings.MEDIA_ROOT, 'uploads', final_filename)
#         os.makedirs(os.path.dirname(final_path), exist_ok=True)
#         shutil.move(file_path, final_path)
#         record.image = f'uploads/{final_filename}'
#         record.save()
#
#         # 清理临时目录
#         shutil.rmtree(temp_dir, ignore_errors=True)
#
#         return {
#             'id': record.id,
#             'filename': uploaded_file.name,
#             'source_zip': source_zip_name,
#             'predicted_class': result['predicted_class'],
#             'predicted_class_display': record.get_predicted_class_display(),
#             'confidence': result['confidence'],
#             'processing_time': result['processing_time'],
#             'top5': result.get('top5_predictions', []),
#             'image_url': record.image.url,
#             'success': True
#         }
#     except Exception as e:
#         return {
#             'filename': uploaded_file.name,
#             'error': str(e),
#             'success': False
#         }
#
#
# def process_zip_file(user, zip_file):
#     """处理ZIP压缩包"""
#     results = []
#     temp_dir = tempfile.mkdtemp()
#     zip_path = os.path.join(temp_dir, zip_file.name)
#     zip_filename = zip_file.name  # 保存ZIP文件名
#
#     try:
#         # 保存ZIP文件
#         with open(zip_path, 'wb+') as destination:
#             for chunk in zip_file.chunks():
#                 destination.write(chunk)
#
#         # 解压ZIP
#         extract_dir = os.path.join(temp_dir, 'extracted')
#         os.makedirs(extract_dir, exist_ok=True)
#
#         with zipfile.ZipFile(zip_path, 'r') as zf:
#             zf.extractall(extract_dir)
#
#         # 遍历解压后的文件
#         from django.core.files.base import ContentFile
#         for root, dirs, files in os.walk(extract_dir):
#             for file in files:
#                 if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                     file_path = os.path.join(root, file)
#
#                     # 创建ContentFile对象
#                     with open(file_path, 'rb') as f:
#                         mock_file = ContentFile(f.read(), name=file)
#                         mock_file.content_type = f'image/{file.split(".")[-1].lower()}'
#
#                         # 传递ZIP文件名作为来源
#                         result = process_single_image(user, mock_file, source_zip_name=zip_filename)
#                         results.append(result)
#
#     except Exception as e:
#         results.append({'filename': zip_file.name, 'error': str(e), 'success': False})
#     finally:
#         shutil.rmtree(temp_dir, ignore_errors=True)
#
#     return results
#
#
# @login_required
# def batch_classify(request):
#     """批量分类视图"""
#     if request.method == 'POST':
#         results = []
#
#         # 处理多文件上传
#         if request.FILES.getlist('images'):
#             uploaded_files = request.FILES.getlist('images')
#
#             for uploaded_file in uploaded_files:
#                 # 验证文件类型
#                 if uploaded_file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
#                     results.append({
#                         'filename': uploaded_file.name,
#                         'error': '不支持的文件格式，请上传JPEG或PNG图片',
#                         'success': False
#                     })
#                     continue
#
#                 result = process_single_image(request.user, uploaded_file)
#                 results.append(result)
#
#         # 处理ZIP压缩包
#         elif request.FILES.get('zip_file'):
#             zip_file = request.FILES['zip_file']
#             results = process_zip_file(request.user, zip_file)
#
#         # 保存结果到session以便显示
#         request.session['batch_results'] = results
#         request.session['batch_total'] = len(results)
#
#         return redirect('classification:batch_result')
#
#     return render(request, 'classification/classify.html')
#
#
# @login_required
# def batch_result(request):
#     """批量分类结果页面"""
#     results = request.session.get('batch_results', [])
#     total = request.session.get('batch_total', 0)
#     success_count = sum(1 for r in results if r.get('success', False))
#     fail_count = total - success_count
#
#     # 统计各类别数量
#     category_stats = {}
#     for r in results:
#         if r.get('success', False) and not r.get('error'):
#             cat = r.get('predicted_class_display', r.get('predicted_class', '未知'))
#             category_stats[cat] = category_stats.get(cat, 0) + 1
#
#     # 排序统计结果
#     category_stats = dict(sorted(category_stats.items(), key=lambda x: x[1], reverse=True))
#
#     context = {
#         'results': results,
#         'total': total,
#         'success_count': success_count,
#         'fail_count': fail_count,
#         'category_stats': category_stats,
#     }
#     return render(request, 'classification/batch_result.html', context)
#
#
# @login_required
# @require_http_methods(["GET"])
# def batch_export(request):
#     """导出批量分类结果"""
#     results = request.session.get('batch_results', [])
#
#     response = HttpResponse(content_type='text/csv; charset=utf-8')
#     response[
#         'Content-Disposition'] = f'attachment; filename="batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
#
#     # 写入BOM头解决中文乱码
#     response.write('\ufeff')
#
#     writer = csv.writer(response)
#     writer.writerow(['序号', '文件名', '来源ZIP', '预测类别', '识别确信度(%)', '处理时间(秒)', '状态', '错误信息'])
#
#     for idx, r in enumerate(results, 1):
#         writer.writerow([
#             idx,
#             r.get('filename', ''),
#             r.get('source_zip', ''),
#             r.get('predicted_class_display', '-'),
#             f"{r.get('confidence', 0):.2f}" if r.get('success') else '-',
#             f"{r.get('processing_time', 0):.3f}" if r.get('success') else '-',
#             '成功' if r.get('success', False) else '失败',
#             r.get('error', '')
#         ])
#
#     return response


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import ClassificationRecord
from .model_loader import get_classifier
import os
from django.conf import settings
import zipfile
import tempfile
import shutil
import csv
from datetime import datetime


@login_required
def index(request):
    recent_records = ClassificationRecord.objects.filter(user=request.user)[:10]
    total_count = ClassificationRecord.objects.filter(user=request.user).count()

    # 计算平均确信度
    from django.db.models import Avg
    avg_result = ClassificationRecord.objects.filter(user=request.user).aggregate(avg=Avg('confidence'))
    avg_confidence = avg_result['avg'] if avg_result['avg'] else 0

    # 统计类别数量
    category_count = ClassificationRecord.objects.filter(user=request.user).values('predicted_class').distinct().count()

    context = {
        'recent_records': recent_records,
        'total_count': total_count,
        'avg_confidence': avg_confidence,
        'category_count': category_count,
    }
    return render(request, 'classification/index.html', context)


@login_required
def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_file = request.FILES['image']

            # 获取用户选择的模型
            model_choice = request.POST.get('model_choice', 'cgfnet')
            print(f"用户选择模型: {model_choice}")

            # 验证文件类型
            allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
            if uploaded_file.content_type not in allowed_types:
                messages.error(request, '请上传JPEG或PNG格式的图片')
                return redirect('classification:classify')

            # 保存临时文件
            file_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # 根据用户选择的模型进行分类
            classifier = get_classifier(model_choice)
            result = classifier.predict(file_path)

            # 保存记录到数据库
            record = ClassificationRecord.objects.create(
                user=request.user,
                image=f'temp/{uploaded_file.name}',
                original_filename=uploaded_file.name,
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                processing_time=result['processing_time'],
                model_name=result.get('model_name', model_choice)  # 保存使用的模型
            )

            # 移动文件到正式目录
            final_path = os.path.join(settings.MEDIA_ROOT, f'uploads/{record.id}_{uploaded_file.name}')
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            os.rename(file_path, final_path)
            record.image = f'uploads/{record.id}_{uploaded_file.name}'
            record.save()

            # 根据确信度添加提示
            confidence_warning = ""
            confidence_level = ""

            if result['confidence'] < 50:
                confidence_warning = "⚠️ 识别确信度较低，建议使用更清晰的图像"
                confidence_level = "低"
            elif result['confidence'] < 70:
                confidence_warning = "ℹ️ 识别确信度中等，结果仅供参考"
                confidence_level = "中等"
            elif result['confidence'] < 85:
                confidence_level = "较高"
            else:
                confidence_level = "极高"

            context = {
                'record': record,
                'result': result,
                'top5': result['top5_predictions'],
                'confidence_warning': confidence_warning,
                'confidence_level': confidence_level
            }
            return render(request, 'classification/result.html', context)

        except FileNotFoundError as e:
            messages.error(request, f'模型文件未找到: {e}')
            return redirect('classification:classify')
        except Exception as e:
            messages.error(request, f'分类失败: {str(e)}')
            return redirect('classification:classify')

    return render(request, 'classification/classify.html')


@login_required
def history(request):
    records = ClassificationRecord.objects.filter(user=request.user)
    paginator = Paginator(records, 12)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    context = {'page_obj': page_obj, 'total_count': records.count()}
    return render(request, 'classification/history.html', context)


@login_required
def record_detail(request, record_id):
    try:
        record = ClassificationRecord.objects.get(id=record_id, user=request.user)
        context = {'record': record}
        return render(request, 'classification/record_detail.html', context)
    except ClassificationRecord.DoesNotExist:
        messages.error(request, '记录不存在')
        return redirect('classification:history')


@login_required
def delete_record(request, record_id):
    if request.method == 'POST':
        try:
            record = ClassificationRecord.objects.get(id=record_id, user=request.user)
            if record.image and os.path.exists(record.image.path):
                os.remove(record.image.path)
            record.delete()
            messages.success(request, '记录已删除')
        except ClassificationRecord.DoesNotExist:
            messages.error(request, '记录不存在')
    return redirect('classification:history')


@login_required
def statistics(request):
    from django.db.models import Count, Avg
    from datetime import datetime, timedelta

    stats = ClassificationRecord.objects.filter(
        user=request.user
    ).values('predicted_class').annotate(
        count=Count('predicted_class')
    ).order_by('-count')[:10]

    # 获取近30天统计
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    daily_stats = ClassificationRecord.objects.filter(
        user=request.user,
        created_at__range=[start_date, end_date]
    ).extra({'date': "DATE(created_at)"}).values('date').annotate(
        count=Count('id')
    ).order_by('date')

    total_count = ClassificationRecord.objects.filter(user=request.user).count()

    if total_count > 0:
        avg_confidence_result = ClassificationRecord.objects.filter(
            user=request.user
        ).aggregate(avg=Avg('confidence'))
        avg_confidence = avg_confidence_result['avg'] if avg_confidence_result['avg'] is not None else 0
    else:
        avg_confidence = 0

    context = {
        'stats': stats,
        'daily_stats': list(daily_stats),
        'total_count': total_count,
        'avg_confidence': avg_confidence
    }
    return render(request, 'classification/statistics.html', context)


@csrf_exempt
@require_http_methods(["POST"])
@login_required
def api_classify(request):
    if request.FILES.get('image'):
        try:
            uploaded_file = request.FILES['image']
            # API默认使用CGFNet，也可以从请求中获取
            model_choice = request.POST.get('model_choice', 'cgfnet')
            classifier = get_classifier(model_choice)

            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            result = classifier.predict(temp_path)
            os.remove(temp_path)

            return JsonResponse({
                'success': True,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'processing_time': result['processing_time'],
                'model_name': result.get('model_name', model_choice)
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': '未提供图像'})


# ==================== 批量分类功能 ====================

def process_single_image(user, uploaded_file, source_zip_name=None, model_choice='cgfnet'):
    """处理单张图片

    Args:
        user: 用户对象
        uploaded_file: 上传的文件
        source_zip_name: 来源ZIP文件名（可选）
        model_choice: 模型选择（'cgfnet' 或 'efficientnet'）
    """
    try:
        # 保存临时文件
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # 根据用户选择的模型进行分类
        classifier = get_classifier(model_choice)
        result = classifier.predict(file_path)

        # 保存记录到数据库
        record = ClassificationRecord.objects.create(
            user=user,
            image='',  # 临时，稍后更新
            original_filename=uploaded_file.name,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            processing_time=result['processing_time'],
            source_zip_name=source_zip_name,
            model_name=result.get('model_name', model_choice)  # 保存使用的模型
        )

        # 移动文件到正式目录
        final_filename = f"{record.id}_{uploaded_file.name}"
        final_path = os.path.join(settings.MEDIA_ROOT, 'uploads', final_filename)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.move(file_path, final_path)
        record.image = f'uploads/{final_filename}'
        record.save()

        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            'id': record.id,
            'filename': uploaded_file.name,
            'source_zip': source_zip_name,
            'predicted_class': result['predicted_class'],
            'predicted_class_display': record.get_predicted_class_display(),
            'confidence': result['confidence'],
            'processing_time': result['processing_time'],
            'top5': result.get('top5_predictions', []),
            'image_url': record.image.url,
            'model_name': result.get('model_name', model_choice),  # 返回模型名称
            'success': True
        }
    except Exception as e:
        return {
            'filename': uploaded_file.name,
            'error': str(e),
            'success': False
        }


def process_zip_file(user, zip_file, model_choice='cgfnet'):
    """处理ZIP压缩包

    Args:
        user: 用户对象
        zip_file: 上传的ZIP文件
        model_choice: 模型选择（'cgfnet' 或 'efficientnet'）
    """
    results = []
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, zip_file.name)
    zip_filename = zip_file.name

    try:
        # 保存ZIP文件
        with open(zip_path, 'wb+') as destination:
            for chunk in zip_file.chunks():
                destination.write(chunk)

        # 解压ZIP
        extract_dir = os.path.join(temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # 遍历解压后的文件
        from django.core.files.base import ContentFile
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)

                    with open(file_path, 'rb') as f:
                        mock_file = ContentFile(f.read(), name=file)
                        mock_file.content_type = f'image/{file.split(".")[-1].lower()}'

                        # 传递ZIP文件名作为来源和模型选择
                        result = process_single_image(user, mock_file, source_zip_name=zip_filename,
                                                      model_choice=model_choice)
                        results.append(result)

    except Exception as e:
        results.append({'filename': zip_file.name, 'error': str(e), 'success': False})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


@login_required
def batch_classify(request):
    """批量分类视图"""
    if request.method == 'POST':
        results = []

        # 获取批量分类的模型选择（从隐藏字段）
        batch_model_choice = request.POST.get('batch_model_choice', 'cgfnet')
        print(f"批量分类使用模型: {batch_model_choice}")

        # 处理多文件上传
        if request.FILES.getlist('images'):
            uploaded_files = request.FILES.getlist('images')

            for uploaded_file in uploaded_files:
                # 验证文件类型
                if uploaded_file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
                    results.append({
                        'filename': uploaded_file.name,
                        'error': '不支持的文件格式，请上传JPEG或PNG图片',
                        'success': False
                    })
                    continue

                result = process_single_image(request.user, uploaded_file, model_choice=batch_model_choice)
                results.append(result)

        # 处理ZIP压缩包
        elif request.FILES.get('zip_file'):
            zip_file = request.FILES['zip_file']
            results = process_zip_file(request.user, zip_file, model_choice=batch_model_choice)

        # 保存结果到session以便显示
        request.session['batch_results'] = results
        request.session['batch_total'] = len(results)

        return redirect('classification:batch_result')

    return render(request, 'classification/classify.html')


@login_required
def batch_result(request):
    """批量分类结果页面"""
    results = request.session.get('batch_results', [])
    total = request.session.get('batch_total', 0)
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = total - success_count

    # 计算平均确信度
    total_confidence = 0
    for r in results:
        if r.get('success', False) and not r.get('error'):
            total_confidence += r.get('confidence', 0)
    avg_confidence = total_confidence / success_count if success_count > 0 else 0

    # 统计各类别数量
    category_stats = {}
    for r in results:
        if r.get('success', False) and not r.get('error'):
            cat = r.get('predicted_class_display', r.get('predicted_class', '未知'))
            category_stats[cat] = category_stats.get(cat, 0) + 1

    # 排序统计结果
    category_stats = dict(sorted(category_stats.items(), key=lambda x: x[1], reverse=True))

    context = {
        'results': results,
        'total': total,
        'success_count': success_count,
        'fail_count': fail_count,
        'avg_confidence': avg_confidence,
        'category_stats': category_stats,
    }
    return render(request, 'classification/batch_result.html', context)


@login_required
@require_http_methods(["GET"])
def batch_export(request):
    """导出批量分类结果"""
    results = request.session.get('batch_results', [])

    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response[
        'Content-Disposition'] = f'attachment; filename="batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'

    # 写入BOM头解决中文乱码
    response.write('\ufeff')

    writer = csv.writer(response)
    writer.writerow(
        ['序号', '文件名', '来源ZIP', '使用模型', '预测类别', '识别确信度(%)', '处理时间(秒)', '状态', '错误信息'])

    for idx, r in enumerate(results, 1):
        writer.writerow([
            idx,
            r.get('filename', ''),
            r.get('source_zip', ''),
            r.get('model_name', '-'),
            r.get('predicted_class_display', '-'),
            f"{r.get('confidence', 0):.2f}" if r.get('success') else '-',
            f"{r.get('processing_time', 0):.3f}" if r.get('success') else '-',
            '成功' if r.get('success', False) else '失败',
            r.get('error', '')
        ])

    return response