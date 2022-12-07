config_preprocessing = dict(
    {
        'credit_g.csv':{
            'numericals':["duration","credit_amount","installment_commitment","residence_since","age","existing_credits","num_dependents"],
            'non_numericals':["checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker"]
            },
        'bank_marketing.csv':{
            'numericals':["V1","V6","V10","V12","V13","V14","V15"],
            'non_numericals':["V2","V3","V4","V5","V7","V8","V9","V11","V16"]
            },
    }
)