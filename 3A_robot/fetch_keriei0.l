(defun keirei (&optional (tm 10000))

  (send *fetch* :angle-vector #f(200.0 -70.0 10.0 4.26256e-14 -100.0 170.0 70.0 -40.0 0.0 0.0))
  (send *ri* :angle-vector (send *fetch* :angle-vector) tm))

(print "load success")
