from crewai import Crew,Process
from agents import blog_researcher,blog_writer,sql_dev,data_analyst,report_writer
from tasks import research_task,write_task,extract_data,analyze_data


# Forming the tech-focused crew with some enhanced configurations
# crew = Crew(
#   agents=[blog_researcher, blog_writer],
#   tasks=[research_task, write_task],
#   process=Process.sequential,  # Optional: Sequential task execution is default
#   memory=True,
#   cache=True,
#   max_rpm=100,
#   share_crew=True
# )


crew = Crew(
    agents=[sql_dev, data_analyst],
    tasks=[ analyze_data],
    process=Process.sequential,
    memory=False,
    output_log_file="crew.log",
)

## start the task execution process with enhanced feedback
result=crew.kickoff(inputs={'query':'how many facility we have facility table ?'})
print("Here is Result: \n")
print(result)