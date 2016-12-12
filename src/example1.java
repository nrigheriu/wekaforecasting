import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.timeseries.eval.TSEvaluation;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


/**
 * Created by cycle on 14.11.16.
 */
public class example1 {
    public static void main(String[] args){
        String separatedRun = "50, 49, 48,  1, 47, 34, 35, 33, 46, 32, 36, 45, 37, 31, 38, 39, 30, 44, 29, 40," +
                    "100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 81, 82, 80, 90, 79, 83, 78, 84, 85, 88";
        String atOnceRun = "100, 99, 98, 97, 96, 95, 94, 93, " +
                "92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76,  1, 75, 74, 73, 72, 71, 70, 69, 68, 64, 63, 66, 67, 65, 62";

        System.out.println(separatedRun.split(",")  );
    }
}
