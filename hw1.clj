(ns hw1
  "Managing and plotting rain gauge data"
  (:use clojure.data.csv)
  (:require csv
            quickthing
            [missionary.core      :as m]
            [thi.ng.geom.viz.core :as viz]
            [thi.ng.geom.svg.core :as svg]
            [tick.core      :as tick]))

(set! *warn-on-reflection* true)

(def
  input
  (atom
    {:file "./data/hw1_train.dat"
     :iterations 1000 }))

(defn
  hw1-data
  []
  (->>
    (->
      (:file
       @input)
      slurp
      (read-csv
        :separator \tab))
    (mapv
      (fn [row]
        (->>
          row
          (mapv
            #(Double/parseDouble
               %)))))))
#_
(hw1-data)

(def
  perceptron
  "Initialized to a vector
  [0.0 ... 0.0]
  of same length as 'x'"
  (atom
    (->
      (hw1-data)
      (nth 0) ;; first
      count
      ;; dec ;; we add a 1.0 to `x_0`
      (repeat
        0.0))))
#_
(identity
  @perceptron)
;; => (0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)

(defn
  num-points-fun
  []
  (->
    (hw1-data)
    count))
#_
(num-points)
;; => 256


(def
  num-points
  (memoize
    num-points-fun))

(defn-
  xs-fun
  []
  (->>
    (hw1-data)
    (mapv
      butlast)
    (mapv
      #(into
         [1.0]
         %))))
#_
(->
  (xs)
  first)
;; => (-46.540307805649675
;;     114.2
;;     17.93
;;     29.06
;;     0.01506
;;     0.002783
;;     0.02163
;;     0.04608
;;     0.2151
;;     0.1251)

(def
  xs
  (memoize
    xs-fun))

(defn-
  ys-fun
  []
  (->>
    (hw1-data)
    (mapv
      last)))
#_
(ys)
;; => [1.0
;;     1.0
;; ....
;;     -1.0
;;     1.0
;;     1.0]

(def
  ys
  (memoize
    ys-fun))

(defn
  sign
  [^Double x]
  (case
      (Math/signum
        x)
    -1.0 -1.0
    1.0  1.0
    0.0  1.0))
#_
(sign 0.0)
;; => 1.0
#_
(sign 999.9)
;; => 1.0
#_
(sign -123.0)
;; => -1.0

(defn
  classified?
  "Test if perceptron `w`
  is classifying vector `x`
  as the correct value `y`"
  [w
   x
   y]
  (let [test-y (reduce ;; an inner product
                 + ;; add terms
                 (mapv
                   * ;; multiply terms
                   w
                   x))]
    (==
      (sign
        y)
      (sign
        test-y))))
#_
(classified?
  '(-3.0 139.55522341694902 -418.6 -83.78999999999999 -126.82999999999998 -0.06325 -0.007968 -0.04735 -0.33099999999999996 -1.1268 -0.535)
  [1 -46.48330780564967 161.1 34.37 88.25 0.03897 0.004445 0.02168 0.106 0.4634 0.2048]
  1.0)


(defn
  misclassified-x
  [w]
  (->>
    (:iterations
     @input)
    range
    (some
      (fn [iteration]
        (let [rand-idx (->>
                         (num-points)
                         rand-int)]
          (let [y (->
                    (ys)
                    (get
                      rand-idx))
                x (->
                    (xs)
                    (get
                      rand-idx))]
            (let [test-y (->> ;; inner product x and w0
                           (mapv
                             *
                             w
                             x) ;; multiply terms
                           (reduce
                             +))] ;; add them up
              (if (classified?
                    w
                    x
                    y)
                nil ;; go again
                {:iteration iteration
                 :point-idx rand-idx
                 :x         x
                 :y         y}))))))))
#_
(misclassified-x)
;; => {:iteration 3,
;;     :point-idx 145,
;;     :x
;;     [1
;;      -46.53140780564967
;;      177.0
;;      32.68
;;      83.16
;;      0.05121
;;      0.005195
;;      0.02175
;;      0.3523
;;      0.7892
;;      0.2733],
;;     :y -1.0}

(defn
  classified-fraction
  [w]
  (/
    (->>
      (mapv
        (fn [x y]
          (if (classified?
                w ;; @perceptron
                x
                y)
            1.0
            0.0))
        (xs)
        (ys))
      (reduce
        +))
    (count
      (xs))))
#_
(classified-fraction)
;; => 0.0

(defn
  update-perceptron
  "Take an input perceptron vector `w`
  Which should be classified as `y`"
  [w
   x
   y]    
  (mapv
    +
    w
    (mapv
      (partial
        *
        y)
      x)))


#_
(let [{:keys [iteration
              point-idx
              x
              y]} (misclassified-x)]
  (if (nil? x)
    (println
      "NOTHING MISCLASSIFIED")
    (println
      "#############"
      "\nSUMMARY"
      "\nPoint X:\n"
      x
      "\nCorrect Classification (Y):\n"
      y
      "\nIndex:\n"
      point-idx
      "\n# of iterations to find\n"
      iteration
      "\nOriginal W:\n"
      @perceptron
      "\nNew W:\n"
      (update-perceptron
        @perceptron
        x
        y))))



(defn
  tune-perceptron
  [iterations
   perceptron]
  (if (zero? iterations)
    (do
      #_
      (println
        "############################\n"
        "RAN OUT OF TUNING STEPS\n"
        "Perceptron is:\n"
        perceptron
        "\nClassified Fraction:\n"
        (classified-fraction
          perceptron))
      [(classified-fraction
         perceptron)
       0
       perceptron])
    (let [{:keys [iteration
                  point-idx
                  x
                  y]} (misclassified-x
                        perceptron)]
      (if (nil? x)
        (do
          #_
          (println
            "############################\n"
            "PERCEPTRON TUNED\n"
            "Perceptron is:\n"
            perceptron
            "\nMissclassified Fraction:\n"
            (classified-fraction
              perceptron))
          [(classified-fraction
             perceptron)
           iterations
           perceptron])
            #_
            (println
            "+++++++++++++++++++++++++++\n"
            "PERCEPTRON UPDATING\n"
            "Old perceptron\n"
            perceptron
            "\nMissclassified X:\n"
            x
            "\nMissclassified Y:\n"
            y
            "\nClassified Fraction:\n"
            (classified-fraction
              perceptron))
          (recur
            (dec
              iterations)
            (update-perceptron
              perceptron
              x
              y))))))

;; ###############################
;; SOLUTIONS
;; ###############################

;; Problem 13
#_
(->>
  1000
  range
  (mapv
    (fn [_]
      (->
        (tune-perceptron
          (*
            256
            1/2)
          @perceptron)
        (nth
          0))))
  (reduce
    +)
  (*
    1/1000))
;; => 0.820609375

;; Problem 14
#_
(->>
  1000
  range
  (mapv
    (fn [_]
      (->
        (tune-perceptron
          (*
            256
            4)
          @perceptron)
        (nth
          0))))
  (reduce
    +)
  (*
    1/1000))
;; => 0.9998203125

;; Problem 15
#_
(-
  1000
  (nth 
    (->>
      1000
      range
      (map
        (fn [_]
          (tune-perceptron
            (*
              256
              4)
            @perceptron)))
      (mapv     ;; at which iteration it stopped 
        second) ;;(out of 1000)
      sort)
    500))
;; => 425


;; Problem 16
#_
(nth 
  (->>
    1000
    range
    (map
      (fn [_]
        (tune-perceptron
          (*
            256
            4)
          @perceptron)))
    (mapv
      #(nth
         %
         2)) ;; get the perceptron
    (mapv
      #(nth
         %
         0)) ;; get the w0
    sort)
  500)
;; => 34.0

;; Problem 17
;;
;; a bit annoying to reimplement so we just mask `xs`
;; (not happy with this implementation..)


(def
  xs
  (memoize
    (fn []
      (->>
        (xs-fun)
        (mapv
          (fn [x]
            (mapv
              #(/
                 %
                 2.0)
              x)))))))
#_
(-
  1000
  (nth 
    (->>
      1000
      range
      (map
        (fn [_]
          (tune-perceptron
            (*
              256
              4)
            @perceptron)))
      (mapv     ;; at which iteration it stopped 
        second) ;;(out of 1000)
      sort)
    500))
;; => 424;; => 428;; => 426;; => 428

;; Problem 18

(def
  xs
  (memoize
    (fn []
      (->>
        (xs-fun)
        (mapv
          (fn [x]
            (assoc
              x
              0
              0.0)))))))
#_
(-
  1000
  (nth 
    (->>
      1000
      range
      (map
        (fn [_]
          (tune-perceptron
            (*
              256
              4)
            @perceptron)))
      (mapv     ;; at which iteration it stopped 
        second) ;;(out of 1000)
      sort)
    500))
;; => 421

;; Problem 19
(def
  xs
  (memoize
    (fn []
      (->>
        (xs-fun)
        (mapv
          (fn [x]
            (assoc
              x
              0
              -1.0)))))))
#_
(nth 
  (->>
    1000
    range
    (map
      (fn [_]
        (tune-perceptron
          (*
            256
            4)
          @perceptron)))
    (mapv
      #(nth
         %
         2)) ;; get the perceptron
    (mapv
      #(nth
         %
         0)) ;; get the w0
    sort)
  500)
;; => -34.0 ;; this is w0
;; so w0*x0 = 34.0


;; Problem 20
(def
  xs
  (memoize
    (fn []
      (->>
        (xs-fun)
        (mapv
          (fn [x]
            (assoc
              x
              0
              0.1126)))))))
#_
(nth 
  (->>
    1000
    range
    (map
      (fn [_]
        (tune-perceptron
          (*
            256
            4)
          @perceptron)))
    (mapv
      #(nth
         %
         2)) ;; get the perceptron
    (mapv
      #(nth
         %
         0)) ;; get the w0
    sort)
  500)
;; => 3.8284000000000007 ;; this is w0
;; so w0*x0 is
(*
  3.824
  0.1126)
;; => 0.4305824
