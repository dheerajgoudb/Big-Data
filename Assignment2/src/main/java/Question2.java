import java.io.IOException;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Question2 {
	
	public static class BusinessMap extends Mapper<LongWritable, Text, Text, Text>
	{
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
		{
			String[] businessData = StringUtils.split(value.toString(), "^");
			
			if (businessData.length == 3)
			{
				if (businessData[1].contains("NY"))
				{
					context.write(new Text(businessData[0]), new Text(businessData[1]));
				}
			}
		}
	}
	
	public static class BusinessReduce extends Reducer<Text, Text, Text, Text>
	{
		
		public void reduce(Text key, Iterable<Text> values, Context context ) throws IOException, InterruptedException
		{
			for (Text val : values)
			{
				context.write(key, val);
			}
		}
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException 
	{
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2)
		{
			System.err.println("Usage: hadoop jar filename.jar Question2 <input path> <output path>");
		}
		
		Job job = Job.getInstance(conf, "Question2");
		job.setJarByClass(Question2.class);
		job.setMapperClass(BusinessMap.class);
		job.setReducerClass(BusinessReduce.class);
		
		job.setOutputKeyClass(Text.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
