few_shots = [
    {'Question' : "How many users we have?",
     'SQLQuery' : "SELECT count(*) FROM users'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "1118"},
    {'Question': "How many facility or clinic we have ?",
     'SQLQuery':"SELECT count(*) FROM facility",
     'SQLResult': "Result of the SQL query",
     'Answer': "62"},
    {'Question': "how many users in Amanora clinics ?",
     'SQLQuery' : """SELECT COUNT(*) FROM users  INNER JOIN facility ON users.facility_id = facility.id WHERE facility.name = 'Amanora'""",
     'SQLResult': "Result of the SQL query",
     'Answer': "5"},
    {'Question': "how many receipt we have for last year?",
     'SQLQuery' : """SELECT COUNT(*) FROM reciept WHERE YEAR(rect_created_date) = YEAR(CURDATE()) - 1""",
     'SQLResult': "Result of the SQL query",
     'Answer': "480"},
#     {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?" ,
#      'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
# (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
# group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
#  """,
#      'SQLResult': "Result of the SQL query",
#      'Answer': "16725.4"} ,
#      {'Question' : "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?" ,
#       'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
#       'SQLResult': "Result of the SQL query",
#       'Answer' : "17462"},
#     {'Question': "How many white color Levi's shirt I have?",
#      'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
#      'SQLResult': "Result of the SQL query",
#      'Answer' : "290"
#      },
#     {'Question': "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
#      'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
# (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Nike' and size="L"
# group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
#  """,
#      'SQLResult': "Result of the SQL query",
#      'Answer' : "290"
#     }
]