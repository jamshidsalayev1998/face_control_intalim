import asyncio
import aiohttp
import time

async def send_request(session, url, payload, token):
    try:
        headers = {'Authorization': f'Bearer {token}'}
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        return {"error": str(e)}

async def main():
    start_time = time.time()
    
    url = "http://213.230.109.85:8555/answer/create/"
    payload = {
        "title": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAo",
        "question_id": 35,
        "is_true": False,
        # "last_name": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAoeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAo"
    }
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxNTI1MzAxOH0.mWPFFWofLJn8GRVqBUO8pj3LPqFAq9mtvmxEunbXRAo"
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(50)
        tasks = []
        for _ in range(100):
            async def bounded_request(sem):
                async with sem:
                    return await send_request(session, url, payload, token)
            tasks.append(bounded_request(semaphore))
        
        responses = await asyncio.gather(*tasks)
        
        success_count = sum(1 for response in responses if "error" not in response)
    
    elapsed_time = time.time() - start_time
    print(f"Successful responses: {success_count} out of 500")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

asyncio.run(main())