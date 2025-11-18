
import { Cog6ToothIcon } from '@heroicons/react/24/solid';

export default function DashboardLoadingSpinner() {
  return (
    <div className="flex min-h-screen w-full">
      <div
        className="flex flex-col justify-center items-center 
        w-full relative bg-linear-to-br from-sky-100 via-sky-200
        to-sky-300 overflow-hidden"
      >
        <div className="mb-6">
          <Cog6ToothIcon 
            className="w-20 h-20 text-sky-600 animate-spin" 
          />
        </div>
        
      </div>

    </div>
  );
}