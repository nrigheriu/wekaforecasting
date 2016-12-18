import weka.core.Instances;
import java.io.*;
import java.util.ArrayList;

import weka.classifiers.functions.LinearRegression;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Utils;

/**
 * Created by cycle on 10.11.16.
 */
public class anotherExample {

    public static void main(String[] args) {
        try {
            ArrayList<Integer> array = new ArrayList<Integer>(3);
            array.add(2);
            array.add(3);
            array.add(4);
            String asd = array.toString();
            System.out.println(array.toString().substring(1, array.toString().length()-1));
        } catch (Exception e) {
            e.printStackTrace();
        }

        // setting class attribute

    }
}
