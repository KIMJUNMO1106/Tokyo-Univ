;; robotの初期化
(load "package://dxl_armed_turtlebot/euslisp/dxl-armed-turtlebot-interface.l")
(dxl-armed-turtlebot-init)

;; jsk_pcl_rosのmsgを使うため、loadする
;;(ros::load-ros-manifest "jsk_pcl_ros")
(ros::roseus-add-msgs "jsk_recognition_msgs")

;; クラスタリング結果であるBoundingBoxのtopic名
;;(defvar *topic-name* "/camera/depth_registered/cluster_decomposer/boxes")
(defvar *topic-name* "/camera/depth_registered/boxes")
(defvar *bounding-box-list* nil)
(setq *recog_botton* nil)
;; ros::initする
(ros::roseus "boundingboxarray_subscriber")
(send *ri* :angle-vector (send *dxl-armed-turtlebot* :angle-vector) 3000 )
(send *ri* :stop-grasp)
;; コールバック関数
(defun bounding-box-array-cb (msg)
  (setq *bounding-box-list* (send msg :boxes)) ;; boxesは、BoundingBoxのArray(Euslispではlist)
  ;; BoundingBoxがあれば表示する
  (when *bounding-box-list*
    (send *irtviewer* :draw-objects :flush nil)
    (mapcar #'(lambda (b)
                ;; BoundingBoxは、dimensions(直方体の幅・奥行き・高さ)をもつ
                (let* ((dims (ros::tf-point->pos (send b :dimensions)))
                       (bx (make-cube (elt dims 0) (elt dims 1) (elt dims 2)))
                       ;; (1) BoundingBoxのカメラ相対の座標系は、geometry_msgs/Poseという型で得られるので、Euslispのcoordsに変換する
                       (cam->obj-coords (ros::tf-pose->coords (send b :pose)))
                       ;; (2) *dxl-armed-turtlebot*モデルがカメラの座標系をもってるので、取得する
                       (cam-coords (send (send *dxl-armed-turtlebot* :camera_rgb_optical_frame_lk) :copy-worldcoords)))
                  ;; (3) Euslisp内部でのworld座標系の値にして、そこにmake-cubeの箱を設置する
                  (send bx :newcoords (send cam-coords :transform cam->obj-coords))
                  (send bx :worldcoords)
                  (send bx :draw-on :flush nil :color #f(1 0 0)) ;; 描画
		  (setq *bbox* bx)
		  (print *bbox*)
		  (send *bbox* :pos)
		  (print (elt (send *bbox* :pos) 0))
                  bx))
    

            *bounding-box-list*)
   
    (send *ri* :go-pos 0 0 (* 60 (/ (elt (send *bbox* :pos) 1) (elt (send *bbox* :pos) 0))))
    (send *ri* :go-pos (*(-(elt (send *bbox* :pos) 0) 500) 0.001) 0 0)
   ;; (send *ri* :go-pos 0.2 0 0)
    
    (send *ri* :start-grasp)
    
    (setq *recog_botton* t)
    (send *irtviewer* :viewer :viewsurface :flush)
    ))

(ros::subscribe *topic-name* jsk_recognition_msgs::BoundingBoxArray #'bounding-box-array-cb 1)

(do-until-key
 (x::window-main-one) ;; IRT viewerの視点を変えられる。見にくければ変えよう
 (if (eq *recog_botton* t) (ros::unsubscribe *topic-name*))
 (ros::spin-once)
 (ros::sleep)
 )
