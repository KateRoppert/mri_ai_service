# ===== БЫСТРАЯ ПРОВЕРКА (выполните эти команды) =====

echo "🔍 Быстрая проверка типа диска..."
echo ""

# 1. Самый простой способ
echo "1️⃣ Основная информация о дисках:"
lsblk -d -o name,size,rota,type,model
echo ""
echo "   ROTA: 0 = SSD, 1 = HDD"
echo ""

# 2. Где расположены ваши данные
echo "2️⃣ Где примонтированы диски:"
df -h /home /tmp / 2>/dev/null | grep -v tmpfs
echo ""

# 3. Проверка конкретного диска где ваши данные
echo "3️⃣ Тип диска для вашей домашней директории:"
home_disk=$(df /home 2>/dev/null | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//' | sed 's|/dev/||')
if [ -n "$home_disk" ]; then
    rota=$(cat /sys/block/$home_disk/queue/rotational 2>/dev/null)
    if [ "$rota" = "0" ]; then
        echo "   /home находится на: SSD ✅"
        if [[ $home_disk == nvme* ]]; then
            echo "   Тип: NVMe SSD (очень быстрый) ⚡"
        else
            echo "   Тип: SATA SSD (быстрый) 🚀"
        fi
    elif [ "$rota" = "1" ]; then
        echo "   /home находится на: HDD 💿"
    else
        echo "   /home: не удалось определить тип"
    fi
else
    echo "   Не удалось определить диск для /home"
fi
echo ""

# 4. Быстрый тест скорости чтения
echo "4️⃣ Быстрый тест скорости (создаём файл 50MB):"
test_file="/tmp/speed_test_$$"
if timeout 10 dd if=/dev/zero of="$test_file" bs=1M count=50 2>/dev/null; then
    sync
    echo "   Тестирование скорости чтения..."
    
    # Тест чтения с выводом скорости
    speed_output=$(timeout 10 dd if="$test_file" of=/dev/null bs=1M 2>&1)
    speed=$(echo "$speed_output" | grep -o '[0-9.]* [GM]B/s' | tail -1)
    
    if [ -n "$speed" ]; then
        echo "   Скорость: $speed"
        
        # Интерпретация
        if echo "$speed" | grep -q "GB/s"; then
            echo "   Результат: Очень быстрый диск (вероятно NVMe SSD) ⚡⚡⚡"
        elif echo "$speed" | grep -o '[0-9]*' | head -1 | awk '$1 > 300 {exit 0} {exit 1}'; then
            echo "   Результат: Быстрый диск (вероятно SATA SSD) ⚡⚡"
        elif echo "$speed" | grep -o '[0-9]*' | head -1 | awk '$1 > 100 {exit 0} {exit 1}'; then
            echo "   Результат: Средняя скорость (SATA SSD или быстрый HDD) ⚡"
        else
            echo "   Результат: Медленный диск (вероятно HDD) 💿"
        fi
    else
        echo "   Не удалось измерить скорость"
    fi
    
    rm -f "$test_file"
else
    echo "   Ошибка тестирования скорости"
fi
echo ""

# 5. Итоговая рекомендация
echo "🎯 РЕКОМЕНДАЦИЯ ДЛЯ ASYNC I/O:"
home_disk=$(df /home 2>/dev/null | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//' | sed 's|/dev/||')
if [ -n "$home_disk" ]; then
    rota=$(cat /sys/block/$home_disk/queue/rotational 2>/dev/null)
    if [ "$rota" = "1" ]; then
        echo "   ✅ СТОИТ использовать async I/O (у вас HDD)"
        echo "   Ожидаемое ускорение: 20-40%"
    elif [ "$rota" = "0" ]; then
        if [[ $home_disk == nvme* ]]; then
            echo "   ❌ async I/O не даст большого эффекта (у вас NVMe SSD)"
            echo "   Ожидаемое ускорение: 3-8%"
        else
            echo "   🟡 async I/O может помочь немного (у вас SATA SSD)"
            echo "   Ожидаемое ускорение: 8-15%"
        fi
    fi
else
    echo "   🤷 Не удалось определить тип диска автоматически"
fi
echo ""

echo "📁 Если ваши данные на сетевом диске (NFS/CIFS), то async I/O точно стоит использовать!"