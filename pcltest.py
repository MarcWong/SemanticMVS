import pcl
p = pcl.load('/data1/Dataset/knn/undistorted/0/dense/model_dense.ply')
fil = p.make_statistical_outlier_filter()
fil.set_mean_k (50)
fil.set_std_dev_mul_thresh (1.0)
pcl.save(fil.filter(),"inliers.ply")