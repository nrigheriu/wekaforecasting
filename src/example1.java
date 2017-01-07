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
        String separatedRun = "172, 173, 171, 174, 175, 170, 176, 169, 168, 177, 167, 178, 166, 179, 165, 164, 180, 121, 122, 163, 123, 120, 124, 181, 125, 119, 162, 126, 118, 127, 117, 128, 129, 131, 130, 161, 133, 132, 136, 134, 135, 182, 116, 137, 138, 192, 160, 139, 115, 183, 114, 191, 140, 159, 184, 113, 190, 141, 185, 112, 189, 158, 142, 188, 186, 187, 111, 143, 157, 110, 144, 109, 156, 145, 108, 155, 146, 107, 147, 154, 148, 153, 106, 149, 152, 150, 151, 105, 96, 104";
        String firstpartexcludedRun = "192, 191, 190, 189, 188, 187, 159, 158, 160, 157, 161, 156, 162, 155, 186, 154, 164, 163, 165, 153, 168, 167, 166, 185, 169, 152, 170, 151, 184, 171, 150, 172, 173, 174, 149, 183, 175, 148, 182, 176, 181, 147, 177, 180, 178, 179, 146, 145, 139, 144, 134, 138, 143, 142, 140, 133, 132, 137, 136, 135, 141, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 104, 103, 105, 107, 106, 102, 98";
        String[] separatedRunList = separatedRun.split(",");
        String[] atOnceList = firstpartexcludedRun.split(",");
        firstpartexcludedRun = firstpartexcludedRun.replace(",", " ");
        separatedRun = separatedRun.replace(",", " ");
        for (String i:separatedRunList){
            if(!firstpartexcludedRun.contains(i)){
                System.out.println("Contained in separated, not in all@Once: " + i);
            }
        }
        for(String i:atOnceList){
            if (!separatedRun.contains(i)) {
                System.out.println("Contained in @once, not in separated" + i);
            }
        }
    }
}
