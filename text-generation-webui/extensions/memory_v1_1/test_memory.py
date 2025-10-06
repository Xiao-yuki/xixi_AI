from memory_system import MemorySystem

memsys = MemorySystem()

while True:
    print("\n==== 記憶系統 ====")
    print("1. 新增記憶")
    print("2. 搜尋記憶")
    print("3. 使用記憶")
    print("4. 執行衰退")
    print("5. 顯示所有記憶")
    print("0. 離開")
    choice = input("選擇操作: ")

    if choice == "1":
        content = input("輸入記憶內容：")
        keywords = input("輸入關鍵字（以逗號分隔）：").split(",")
        preference = float(input("喜好程度（-1 到 1）："))
        memsys.add_memory(content, [k.strip() for k in keywords], preference)
        print("已新增記憶。")

    elif choice == "2":
        q = input("輸入搜尋關鍵字：").split(",")
        results = memsys.search_memory([k.strip() for k in q])
        if not results:
            print("沒有找到相關記憶。")
        else:
            print("搜尋結果：")
            for i, m in enumerate(results):
                print(f"{i+1}. {m['content']} (freq={m['frequency']:.2f}, pref={m['preference']:.2f}, last_used={m['last_used']})")

    elif choice == "3":
        memsys.print_memory()
        mid = input("輸入要使用的記憶 ID：").strip()
        result = memsys.use_memory(mid)
        if result:
            print("已使用記憶並更新。")
        else:
            print("沒有找到這個 ID。")

    elif choice == "4":
        memsys.decay_memory()
        print("已執行衰退與清理。")

    elif choice == "5":
        memsys.print_memory()

    elif choice == "0":
        print("再見！")
        break
