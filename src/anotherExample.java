import weka.core.Instances;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Hashtable;
import java.util.List;
import java.util.Random;
import javax.swing.*;

import weka.classifiers.functions.LinearRegression;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Utils;

/**
 * Created by cycle on 10.11.16.
 */
public class anotherExample {
    public static void main(String[] args){
        String asd = "RES 2.4";
        Scanner sc = new Scanner(asd);
        while (!sc.hasNextDouble())
        {
            sc.next();
        }
        double var1 = sc.nextDouble();
        System.out.println(var1);
    }
}
