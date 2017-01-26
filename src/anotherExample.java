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
    public class Graph extends JPanel {
        public Graph() {
            setSize(500, 500);
        }

        @Override
        public void paintComponent(Graphics g) {
            Graphics2D gr = (Graphics2D) g; // This is if you want to use Graphics2D
            // Now do the drawing here
            ArrayList<Integer> scores = new ArrayList<Integer>(10);

            Random r = new Random();

            for (int i : scores) {
                i = r.nextInt(20);
                System.out.println(r);
            }

            int y1;
            int y2;

            for (int i = 0; i < scores.size() - 1; i++) {
                y1 = (scores.get(i)) * 10;
                y2 = (scores.get(i + 1)) * 10;
                gr.drawLine(i * 10, y1, (i + 1) * 10, y2);
            }
        }
    }

    public static void main(String[] args) {
        Random r = new Random();
        try {
            for (int i = 2; i < 102-2; i++) {
                int chance = r.nextInt(100);
                if(chance < 0)
                    System.out.println("Yes");
                else
                    System.out.println("No");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    protected static BitSet changeBits(int numAttribs, BitSet bitSet){
        Random r = new Random();
        for (int i = 0; i < numAttribs; i++) {
            float chance = r.nextFloat();
            if(chance <= 0.02f){
                if(bitSet.get(i))
                    bitSet.set(i, false);
                else
                    bitSet.set(i, true);
            }
        }
        return bitSet;
    }
    protected static boolean includesMoreThan25PercentOfFeatures(BitSet bitSet, int numAttribs){
        int trueBits = 0;
        for (int i = 0; i < numAttribs; i++) {
            if(bitSet.get(i))
                trueBits++;
        }
        if((float)trueBits/numAttribs > 0.25)
            return true;
        return  false;
    }
    protected static BitSet getStartSet (int numAttribs, int setPercentage){
        BitSet bitSet = new BitSet(numAttribs);
        Random r = new Random();
        for (int i = 0; i < numAttribs; i++) {
            float chance = r.nextFloat();
            if (chance <= 0.25f) {
                bitSet.set(i);
            }
        }
        return bitSet;
    }

}
